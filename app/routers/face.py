import cv2
import logging
import numpy as np
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query

logger = logging.getLogger("uvicorn.error")
from app.services.face_analysis import FaceAnalysisService
from app.services.liveness import LivenessService, LivenessUnavailableError
from app.schemas.face import (
    FaceDetectionResponse, BoundingBox, FaceLandmark,
    FaceEmbeddingResponse, VerifyEmbeddingsRequest, VerifyResponse,
    SearchQueryRequest, SearchResponse, SearchMatch,
    VerifyEmployeeResponse, RegisterFaceResponse,
    RegistrationStatusResponse, DeleteRegistrationResponse,
    FaceRegistrationInfo, SecureVerifyEmployeeResponse,
)
from app.config import settings
from app.dependencies import require_face_api_key

from datetime import datetime, timezone
from app.database import db, require_database

router = APIRouter(prefix="/face", tags=["Face Operations"])
service = FaceAnalysisService()


def get_liveness_service() -> LivenessService:
    """Construct lazily so an unprovisioned model fails per request, not at import."""
    return LivenessService()


def reason_error(status_code: int, reason_code: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"reason_code": reason_code})

def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    return img

def format_detection_response(face) -> FaceDetectionResponse:
    # face.bbox is [x1, y1, x2, y2]
    bbox = BoundingBox(
        x1=float(face.bbox[0]),
        y1=float(face.bbox[1]),
        x2=float(face.bbox[2]),
        y2=float(face.bbox[3])
    )
    # face.kps is 5x2 array
    landmarks = [FaceLandmark(x=float(kp[0]), y=float(kp[1])) for kp in face.kps]
    
    # gender: face.gender is 0 (F) or 1 (M) or vice-versa, or string in some versions
    gender_str = "M" if getattr(face, "gender", 0) == 1 else "F"
    
    return FaceDetectionResponse(
        bbox=bbox,
        confidence=float(face.det_score),
        landmarks=landmarks,
        gender=gender_str,
        age=float(getattr(face, "age", 0.0))
    )

@router.post("/detect", response_model=List[FaceDetectionResponse])
async def detect_faces(file: UploadFile = File(...)):
    """
    Detects all faces in the uploaded image, returning their bounding boxes, landmarks, gender, and age.
    """
    contents = await file.read()
    logger.info(f"Detecting faces in uploaded file: {file.filename} ({len(contents)} bytes)")
    img = decode_image(contents)
    faces = service.detect_faces(img)
    logger.info(f"Detected {len(faces)} face(s) in {file.filename}")
    return [format_detection_response(face) for face in faces]

@router.post("/embedding", response_model=List[FaceEmbeddingResponse])
async def extract_embeddings(
    file: UploadFile = File(...),
    largest_only: bool = Query(True, description="If True, only return the embedding for the largest face detected")
):
    """
    Extracts the 512-dimensional face embedding for face(s) in the uploaded image.
    """
    contents = await file.read()
    logger.info(f"Extracting embedding from file: {file.filename} ({len(contents)} bytes), largest_only={largest_only}")
    img = decode_image(contents)
    faces = service.detect_faces(img)
    
    if not faces:
        logger.warning(f"Embedding extraction failed: No faces detected in {file.filename}")
        raise HTTPException(status_code=404, detail="No faces detected in the image")
        
    if largest_only:
        # Sort and pick the largest face by bounding box area
        largest_face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
        faces_to_process = [largest_face]
    else:
        faces_to_process = faces

    response = []
    for face in faces_to_process:
        bbox = BoundingBox(
            x1=float(face.bbox[0]),
            y1=float(face.bbox[1]),
            x2=float(face.bbox[2]),
            y2=float(face.bbox[3])
        )
        response.append(FaceEmbeddingResponse(
            bbox=bbox,
            embedding=face.embedding.tolist()
        ))
    return response

@router.post("/verify-images", response_model=VerifyResponse)
async def verify_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Detects the largest face in each of the two uploaded images and compares them.
    """
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    img1 = decode_image(contents1)
    img2 = decode_image(contents2)
    
    face1 = service.get_largest_face(img1)
    face2 = service.get_largest_face(img2)
    
    if not face1:
        raise HTTPException(status_code=400, detail="No face detected in Image 1")
    if not face2:
        raise HTTPException(status_code=400, detail="No face detected in Image 2")
        
    similarity = service.compute_similarity(face1.embedding, face2.embedding)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold
    
    return VerifyResponse(
        similarity=similarity,
        verified=verified,
        threshold=active_threshold
    )

@router.post("/verify-embeddings", response_model=VerifyResponse)
async def verify_embeddings(
    request: VerifyEmbeddingsRequest,
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Compares two pre-extracted 512-dimensional face embeddings.
    """
    emb1 = np.array(request.embedding1, dtype=np.float32)
    emb2 = np.array(request.embedding2, dtype=np.float32)
    
    if emb1.shape != (512,) or emb2.shape != (512,):
        raise HTTPException(status_code=400, detail="Embeddings must be 512-dimensional vectors")
        
    similarity = service.compute_similarity(emb1, emb2)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold
    
    return VerifyResponse(
        similarity=similarity,
        verified=verified,
        threshold=active_threshold
    )

# ── Shared registry operations (used by API-key and admin session routes) ──

async def register_face_from_image(user_id: str, contents: bytes) -> RegisterFaceResponse:
    """Detect exactly one face in the image and upsert its embedding.

    Preserves created_at on re-registration via $setOnInsert.
    """
    logger.info(f"Start face registration from image for user_id: {user_id} ({len(contents)} bytes)")
    database = require_database()
    try:
        img = decode_image(contents)
    except HTTPException as exc:
        logger.warning(f"Registration failed for user_id: {user_id} - invalid_image")
        raise reason_error(400, "invalid_image") from exc
    faces = service.detect_faces(img)
    if not faces:
        logger.warning(f"Registration failed for user_id: {user_id} - no_face")
        raise reason_error(400, "no_face")
    if len(faces) > 1:
        logger.warning(f"Registration failed for user_id: {user_id} - multiple_faces ({len(faces)} detected)")
        raise reason_error(400, "multiple_faces")

    face = faces[0]
    try:
        liveness = get_liveness_service().analyze(img, face.bbox)
    except LivenessUnavailableError as exc:
        logger.exception(
            "Liveness model unavailable during face registration for user_id: %s",
            user_id,
        )
        raise reason_error(503, "model_unavailable") from exc
    if not liveness.live:
        logger.warning(f"Registration failed for user_id: {user_id} - spoof_detected (score: {liveness.score:.4f}, threshold: {liveness.threshold})")
        raise reason_error(400, "spoof_detected")

    now = datetime.now(timezone.utc)
    result = await database.face_registry.update_one(
        {"user_id": user_id},
        {
            "$set": {"embedding": face.embedding.tolist(), "updated_at": now},
            "$setOnInsert": {"user_id": user_id, "created_at": now},
        },
        upsert=True,
    )
    doc = await database.face_registry.find_one({"user_id": user_id})
    created = getattr(result, "upserted_id", None) is not None
    logger.info(f"Successfully registered face for user_id: {user_id} | Created: {created}")
    return RegisterFaceResponse(
        user_id=user_id,
        created=created,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )

async def get_registration_status(user_id: str) -> RegistrationStatusResponse:
    database = require_database()
    doc = await database.face_registry.find_one({"user_id": user_id})
    if doc is None:
        return RegistrationStatusResponse(user_id=user_id, registered=False)
    return RegistrationStatusResponse(
        user_id=user_id,
        registered=True,
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )

async def delete_registration(user_id: str) -> DeleteRegistrationResponse:
    database = require_database()
    result = await database.face_registry.delete_one({"user_id": user_id})
    return DeleteRegistrationResponse(user_id=user_id, deleted=result.deleted_count > 0)

async def list_registrations() -> List[FaceRegistrationInfo]:
    database = require_database()
    items: List[FaceRegistrationInfo] = []
    cursor = database.face_registry.find({}, {"user_id": 1, "created_at": 1, "updated_at": 1})
    async for doc in cursor:
        items.append(FaceRegistrationInfo(
            user_id=doc["user_id"],
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        ))
    items.sort(key=lambda x: x.user_id)
    return items

# ── API-key protected registry endpoints ──

@router.post("/register", response_model=RegisterFaceResponse, dependencies=[Depends(require_face_api_key)])
async def register_face(
    user_id: str = Form(..., description="Unique employee identifier"),
    file: UploadFile = File(..., description="Image containing exactly one face"),
):
    """
    Registers (or re-registers) a user's face from an uploaded image.
    Requires the X-API-Key header. Embeddings are never returned.
    """
    contents = await file.read()
    return await register_face_from_image(user_id, contents)

@router.get("/register/{user_id}", response_model=RegistrationStatusResponse, dependencies=[Depends(require_face_api_key)])
async def registration_status(user_id: str):
    """
    Returns whether the user has a registered face (metadata only, no embedding).
    """
    return await get_registration_status(user_id)

@router.delete("/register/{user_id}", response_model=DeleteRegistrationResponse, dependencies=[Depends(require_face_api_key)])
async def remove_registration(user_id: str):
    """
    Deletes the user's face registration. Returns deleted=false when absent.
    """
    return await delete_registration(user_id)

@router.post("/search", response_model=SearchResponse)
async def search_face(
    request: SearchQueryRequest,
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Searches the MongoDB registry for matching faces using the query embedding.
    """
    query_emb = np.array(request.embedding, dtype=np.float32)
    if query_emb.shape != (512,):
        raise HTTPException(status_code=400, detail="Query embedding must be a 512-dimensional vector")
        
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    matches = []
    
    # Query all records from MongoDB face_registry
    cursor = db.face_registry.find({}, {"user_id": 1, "embedding": 1})
    async for doc in cursor:
        reg_emb = np.array(doc["embedding"], dtype=np.float32)
        sim = service.compute_similarity(query_emb, reg_emb)
        if sim >= active_threshold:
            matches.append(SearchMatch(user_id=doc["user_id"], similarity=sim))
            
    # Sort matches by similarity descending
    matches.sort(key=lambda x: x.similarity, reverse=True)
    
    return SearchResponse(matches=matches[:request.limit])

@router.post("/verify-employee", response_model=VerifyEmployeeResponse)
async def verify_employee(
    user_id: str = Form(..., description="Mã nhân viên cần xác thực"),
    file: UploadFile = File(..., description="Ảnh chụp khuôn mặt để đối sánh"),
    threshold: float = Query(None, description="Ngưỡng xác thực tùy chỉnh (Mặc định: 0.45)")
):
    """
    Xác thực khuôn mặt nhân viên từ ảnh tải lên so với dữ liệu đã lưu trong MongoDB.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    # 1. Tìm kiếm thông tin đăng ký của nhân viên trong MongoDB
    doc = await db.face_registry.find_one({"user_id": user_id})
    if not doc:
        return VerifyEmployeeResponse(
            verified=False,
            reason="Nhân viên chưa đăng ký thông tin khuôn mặt trên hệ thống.",
            similarity=None
        )

    registered_emb = np.array(doc["embedding"], dtype=np.float32)

    # 2. Giải mã file ảnh tải lên
    contents = await file.read()
    try:
        img = decode_image(contents)
    except HTTPException as e:
        return VerifyEmployeeResponse(
            verified=False,
            reason="File ảnh không hợp lệ hoặc bị lỗi định dạng.",
            similarity=None
        )

    # 3. Phát hiện khuôn mặt lớn nhất trong ảnh chụp mới
    face = service.get_largest_face(img)
    if not face:
        return VerifyEmployeeResponse(
            verified=False,
            reason="Không phát hiện thấy khuôn mặt nào trong ảnh chụp.",
            similarity=None
        )

    # 4. Tính toán độ tương đồng giữa khuôn mặt chụp và khuôn mặt đăng ký
    similarity = service.compute_similarity(registered_emb, face.embedding)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold

    # 5. Trả về kết quả
    if verified:
        reason = "Xác thực khuôn mặt thành công."
    else:
        reason = f"Xác thực thất bại. Khuôn mặt không khớp với nhân viên đã đăng ký (Độ khớp: {similarity*100:.1f}% < {active_threshold*100:.1f}%)."

    return VerifyEmployeeResponse(
        verified=verified,
        reason=reason,
        similarity=similarity
    )
@router.post(
    "/verify-employee-secure",
    response_model=SecureVerifyEmployeeResponse,
    dependencies=[Depends(require_face_api_key)],
)
async def verify_employee_secure(
    user_id: str = Form(..., description="Unique employee identifier"),
    file: UploadFile = File(..., description="Image containing exactly one face"),
):
    database = require_database()
    doc = await database.face_registry.find_one({"user_id": user_id})
    if doc is None:
        return SecureVerifyEmployeeResponse(
            registered=False,
            face_verified=False,
            similarity=None,
            face_threshold=settings.SIMILARITY_THRESHOLD,
            live=False,
            liveness_score=None,
            liveness_threshold=settings.LIVENESS_THRESHOLD,
            reason_code="not_registered",
        )

    contents = await file.read()
    try:
        img = decode_image(contents)
    except HTTPException as exc:
        raise reason_error(400, "invalid_image") from exc

    faces = service.detect_faces(img)
    if not faces:
        raise reason_error(400, "no_face")
    if len(faces) > 1:
        raise reason_error(400, "multiple_faces")

    face = faces[0]
    try:
        liveness = get_liveness_service().analyze(img, face.bbox)
    except LivenessUnavailableError as exc:
        logger.exception(
            "Liveness model unavailable during secure verification for user_id: %s",
            user_id,
        )
        raise reason_error(503, "model_unavailable") from exc

    if not liveness.live:
        return SecureVerifyEmployeeResponse(
            registered=True,
            face_verified=False,
            similarity=None,
            face_threshold=settings.SIMILARITY_THRESHOLD,
            live=False,
            liveness_score=liveness.score,
            liveness_threshold=liveness.threshold,
            reason_code="spoof_detected",
        )

    registered_embedding = np.asarray(doc["embedding"], dtype=np.float32)
    similarity = service.compute_similarity(registered_embedding, face.embedding)
    face_verified = similarity >= settings.SIMILARITY_THRESHOLD
    return SecureVerifyEmployeeResponse(
        registered=True,
        face_verified=face_verified,
        similarity=similarity,
        face_threshold=settings.SIMILARITY_THRESHOLD,
        live=True,
        liveness_score=liveness.score,
        liveness_threshold=liveness.threshold,
        reason_code="verified" if face_verified else "face_mismatch",
    )
