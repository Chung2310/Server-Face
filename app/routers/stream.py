"""Continuous 1:N face identification over WebSocket.

A client holds one connection open and pushes JPEG frames; the service answers
each frame with who it thinks the person is - no user_id is supplied, the
identity is looked up against the whole registry.

Two properties matter more than raw throughput here:

* **Declining is cheap, guessing is not.** A wrong identification clocks in the
  wrong employee. Every uncertain frame comes back as `unknown` or `ambiguous`.
* **Frames are dropped, never queued.** If inference cannot keep up, the newest
  frame wins and older ones are discarded. Queueing would make answers drift
  further and further behind what the camera is actually seeing.
"""

import asyncio
import hmac
import json
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.services.erp_bridge import ErpBridge
from app.services.face_analysis import FaceAnalysisService
from app.services.face_index import (
    STATUS_EMPTY_REGISTRY,
    FaceIndex,
    IdentityTracker,
    normalize,
)
from app.services.liveness import LivenessService, LivenessUnavailableError

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/face", tags=["Face Stream"])

service = FaceAnalysisService()

# Close codes (4000-4999 is the application-defined range).
CLOSE_UNAUTHORIZED = 4401
CLOSE_DISABLED = 4503

STATUS_NO_FACE = "no_face"
STATUS_LOW_QUALITY = "low_quality"
STATUS_SPOOF = "spoof"
STATUS_ERROR = "error"


def _authorized(websocket: WebSocket) -> bool:
    """Timing-safe API key check, mirroring app.dependencies.require_face_api_key.

    The key is read from a header or the WebSocket subprotocol - never from the
    query string, which lands in access logs and browser history.
    """
    configured = (settings.FACE_API_KEY or "").strip()
    if not configured:
        return False

    presented = websocket.headers.get("x-api-key")
    if not presented:
        # Browsers cannot set headers on WebSocket handshakes, so allow the key
        # to arrive as a subprotocol token instead.
        raw = websocket.headers.get("sec-websocket-protocol", "")
        for token in (part.strip() for part in raw.split(",")):
            if token.startswith("apikey."):
                presented = token[len("apikey."):]
                break

    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), configured.encode("utf-8"))


def _decode_frame(payload: bytes) -> Optional[np.ndarray]:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def _bbox_dict(bbox) -> dict:
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    return {
        "x1": float(values[0]),
        "y1": float(values[1]),
        "x2": float(values[2]),
        "y2": float(values[3]),
    }


def _identify_image(image: np.ndarray, index: FaceIndex) -> dict:
    """Run detection + identification on one frame. Blocking; runs in a thread."""
    face = service.get_largest_face(image)
    if face is None:
        return {"status": STATUS_NO_FACE}

    bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
    width = float(bbox[2] - bbox[0])
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    if width < settings.IDENTIFY_MIN_FACE_PX or det_score < settings.IDENTIFY_MIN_DET_SCORE:
        # Too small or too uncertain to identify reliably - usually someone
        # walking past in the background.
        return {"status": STATUS_LOW_QUALITY, "bbox": _bbox_dict(bbox), "det_score": det_score}

    result = index.identify(normalize(face.embedding))
    return {
        "status": result.status,
        "user_id": result.user_id,
        "similarity": result.similarity,
        "margin": result.margin,
        "threshold": result.threshold,
        "bbox": _bbox_dict(bbox),
        "det_score": det_score,
        "_face": face,
        "_image": image,
    }


def _check_liveness(image: np.ndarray, bbox) -> Optional[bool]:
    """Passive liveness for a committed identity. None means 'could not tell'."""
    if not settings.STREAM_LIVENESS_ENABLED:
        return None
    try:
        return bool(LivenessService().analyze(image, bbox).live)
    except LivenessUnavailableError:
        logger.warning("Liveness unavailable during stream identification")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Liveness failed during stream identification: %s", type(exc).__name__)
        return None


async def _log_stream_identification(user_id: str, similarity: float, live: Optional[bool]) -> None:
    from app.database import get_db

    database = get_db()
    if database is None:
        return
    from datetime import datetime, timezone

    try:
        await database.verification_logs.insert_one({
            "user_id": user_id,
            "verified": True,
            "similarity": similarity,
            "reason": "Nhận diện tự động qua luồng video.",
            "source": "stream",
            "live": live,
            "timestamp": datetime.now(timezone.utc),
            "ip_address": None,
            "device_info": "stream",
        })
    except Exception as e:
        logger.warning(f"Failed to write stream log for user_id={user_id}: {type(e).__name__}")


@router.websocket("/stream")
async def identify_stream(websocket: WebSocket):
    """Identify whoever appears in the incoming frames, continuously.

    Send: binary JPEG frames, or JSON ``{"type": "embedding", "vector": [...512]}``
    for clients that already run the model themselves.

    Receive, per frame::

        {"type": "identify", "status": "identified", "user_id": "NV001",
         "similarity": 0.72, "margin": 0.19, "committed": true, "live": true,
         "bbox": {...}, "frame_id": 42}

    ``committed`` is true only on the frame where the identity is first accepted
    (after the consecutive-frame streak and outside the cooldown window). That is
    the frame that produces an attendance event.
    """
    if not settings.STREAM_ENABLED:
        await websocket.close(code=CLOSE_DISABLED)
        return
    if not _authorized(websocket):
        # Reject before accepting - no frames are ever read from an unauthorised peer.
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return

    # Echo the subprotocol back when the key arrived that way, or the browser
    # aborts the handshake.
    raw_protocols = websocket.headers.get("sec-websocket-protocol", "")
    subprotocol = next(
        (p.strip() for p in raw_protocols.split(",") if p.strip().startswith("apikey.")),
        None,
    )
    await websocket.accept(subprotocol=subprotocol)

    index = FaceIndex()
    await index.ensure_loaded()
    tracker = IdentityTracker()
    bridge = ErpBridge()
    loop = asyncio.get_running_loop()

    frame_id = 0
    logger.info("Identification stream opened (%d enrolled users)", index.size)

    # Single-slot mailbox: the reader always overwrites, so while one frame is
    # being processed any frames that arrive collapse to just the newest one.
    # Without a separate reader the socket buffer would keep old frames and every
    # answer would fall further behind the camera.
    slot: dict = {"message": None}
    arrived = asyncio.Event()
    closed = asyncio.Event()

    async def reader():
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                slot["message"] = message
                arrived.set()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed.set()
            arrived.set()

    reader_task = asyncio.create_task(reader(), name="stream-reader")

    try:
        while True:
            await arrived.wait()
            arrived.clear()
            if closed.is_set() and slot["message"] is None:
                break

            message, slot["message"] = slot["message"], None
            if message is None:
                if closed.is_set():
                    break
                continue

            payload = message.get("bytes")
            text = message.get("text")

            embedding = None
            image = None

            if payload is not None:
                if len(payload) > settings.STREAM_MAX_FRAME_BYTES:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "status": STATUS_ERROR, "reason": "frame_too_large"}
                    ))
                    continue
                image = _decode_frame(payload)
                if image is None:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "status": STATUS_ERROR, "reason": "invalid_frame"}
                    ))
                    continue
            elif text is not None:
                try:
                    request = json.loads(text)
                    vector = np.asarray(request["vector"], dtype=np.float32)
                except (TypeError, ValueError, KeyError):
                    await websocket.send_text(json.dumps(
                        {"type": "error", "status": STATUS_ERROR, "reason": "invalid_message"}
                    ))
                    continue
                if vector.shape != (512,):
                    await websocket.send_text(json.dumps(
                        {"type": "error", "status": STATUS_ERROR, "reason": "invalid_embedding"}
                    ))
                    continue
                embedding = vector
            else:
                continue

            frame_id += 1
            await index.ensure_loaded()
            if embedding is not None:
                result = index.identify(normalize(embedding))
                outcome = {
                    "status": result.status,
                    "user_id": result.user_id,
                    "similarity": result.similarity,
                    "margin": result.margin,
                    "threshold": result.threshold,
                }
            else:
                # ONNX inference is blocking - keep it off the event loop.
                outcome = await loop.run_in_executor(None, _identify_image, image, index)

            face = outcome.pop("_face", None)
            source_image = outcome.pop("_image", None)

            committed_user = None
            live = None
            if outcome["status"] not in (STATUS_NO_FACE, STATUS_LOW_QUALITY, STATUS_EMPTY_REGISTRY):
                from app.services.face_index import IdentifyResult

                committed_user = tracker.observe(
                    IdentifyResult(
                        status=outcome["status"],
                        user_id=outcome.get("user_id"),
                        similarity=outcome.get("similarity"),
                    )
                )

            if committed_user and face is not None and source_image is not None:
                # Liveness only once an identity is committed - running it on every
                # frame would burn CPU for frames that name nobody.
                live = await loop.run_in_executor(
                    None, _check_liveness, source_image, face.bbox
                )
                if live is False:
                    outcome["status"] = STATUS_SPOOF
                    outcome["user_id"] = None
                    tracker.reset()
                    committed_user = None

            if committed_user:
                similarity = float(outcome.get("similarity") or 0.0)
                bridge.publish_identification(committed_user, similarity, live)
                await _log_stream_identification(committed_user, similarity, live)
                logger.info(
                    "Stream identified user_id=%s similarity=%.4f live=%s",
                    committed_user, similarity, live,
                )

            await websocket.send_text(json.dumps({
                "type": "identify",
                "frame_id": frame_id,
                "committed": committed_user is not None,
                "live": live,
                **outcome,
            }))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Identification stream error: %s", type(exc).__name__)
    finally:
        reader_task.cancel()
        await asyncio.gather(reader_task, return_exceptions=True)
        logger.info("Identification stream closed after %d frames", frame_id)
