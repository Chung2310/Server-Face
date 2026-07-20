# Video Liveness Challenge Design

## Goal

Add a complete backend contract for random, one-time video liveness challenges. The InsightFace service accepts WebM and MP4 clips, combines passive anti-spoof inference with ordered facial motion, returns a normalized `liveness_score`, and never retains uploaded biometric media.

## Scope

This change is limited to the current InsightFace service. ERP camera capture and UI integration follow later against the new API contract.

## API

Both endpoints require the existing `X-API-Key` integration authentication.

`POST /api/v1/face/liveness/challenges` creates a challenge and returns:

```json
{
  "challenge_id": "uuid",
  "action": "turn_left",
  "expires_at": "2026-07-20T00:00:00Z"
}
```

The action is selected cryptographically at random from `turn_left`, `turn_right`, and `blink`.

`POST /api/v1/face/liveness/verify-video` accepts multipart fields `challenge_id` and `file`. The file may be WebM or MP4. A successful response returns:

```json
{
  "verified": true,
  "liveness_score": 0.93,
  "passive_score": 0.96,
  "motion_score": 0.93,
  "reason_code": "verified"
}
```

Business verification failures return the same response shape with `verified=false`, relevant nullable scores, and a stable reason code. Invalid input and unavailable dependencies use appropriate 4xx or 503 status codes with a stable `detail.reason_code`.

## Challenge State

MongoDB collection `liveness_challenges` stores `challenge_id`, `action`, `created_at`, `expires_at`, and nullable `used_at`. Challenges expire after 60 seconds and have a TTL index on `expires_at` with `expireAfterSeconds=0`.

Verification atomically changes `used_at` from null before processing. This guarantees that concurrent or repeated submissions cannot reuse a challenge. An attempted verification consumes the challenge even if the uploaded video is malformed or fails liveness.

No video, decoded frame, face crop, or embedding is stored in MongoDB.

## Video Processing

- Maximum upload size: 10 MB.
- Maximum decoded duration: 5 seconds.
- Minimum valid sampled frames: 12.
- Sampling rate cap: approximately 15 frames per second.
- Exactly one face must be present in every sampled frame.
- A face disappearing after detection is `face_lost`; multiple faces are rejected.
- Upload bytes are written to a uniquely named temporary file so OpenCV can decode WebM and MP4. Cleanup runs in `finally` on every outcome.

Every sampled frame runs the existing passive liveness model. The aggregate `passive_score` is a conservative lower percentile of per-frame live probabilities rather than an average.

## Motion Challenges

Turn challenges use yaw from the face pose estimate and require an ordered neutral-to-turn-to-neutral sequence. The turn must deviate at least 15 degrees from the neutral baseline in the requested direction.

Blink uses eye-aspect ratio computed from the standard eye points in the available 68-point landmarks and requires an ordered open-to-closed-to-open sequence.

Each detector produces a normalized `motion_score` between 0 and 1. The final score is:

```text
liveness_score = min(passive_score, motion_score)
```

Verification succeeds only when passive and motion scores independently meet their server-owned thresholds. Client-controlled thresholds are not accepted.

## Reason Codes

Stable codes are `verified`, `challenge_not_found`, `challenge_expired`, `challenge_used`, `invalid_video`, `video_too_large`, `video_too_short`, `video_too_long`, `no_face`, `multiple_faces`, `face_lost`, `model_unavailable`, `spoof_detected`, and `challenge_failed`.

## Security and Privacy

- Challenge actions are unpredictable, short-lived, atomic, and single-use.
- The API fails closed on decoding, face analysis, pose, landmark, passive model, database, or malformed-output failures.
- Logs may contain challenge IDs, actions, reason codes, durations, frame counts, and numeric scores. They must not contain media bytes, crops, embeddings, filesystem paths, API keys, or stack traces returned to clients.
- Temporary video files are deleted after processing and never become audit evidence.

## Testing

Automated tests cover API-key enforcement, action selection, 60-second expiry, TTL index creation, atomic one-time consumption, concurrent reuse, WebM/MP4 decoding contracts, size/duration/frame limits, temporary-file cleanup, exact-one-face continuity, ordered left/right turns, ordered blinking, passive aggregation, composite scoring, unavailable models, stable response codes, and preservation of all existing tests.

Video decoder tests use controlled capture doubles for deterministic behavioral coverage. A local codec smoke test verifies that the installed OpenCV build can open representative WebM and MP4 fixtures when codecs are available.

## Delivery

Implementation follows test-driven development, runs focused and complete verification, and is committed and pushed to `origin/develop`.
