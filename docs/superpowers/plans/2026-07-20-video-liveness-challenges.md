# Video Liveness Challenges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one-time MongoDB-backed video challenges that combine passive and ordered motion scores for WebM/MP4 uploads.

**Architecture:** A focused `VideoLivenessService` owns decoding, sampling, face continuity, passive aggregation, and motion scoring. Router helpers own challenge issuance/atomic consumption and expose authenticated endpoints. MongoDB stores only short-lived challenge metadata.

**Tech Stack:** FastAPI, Motor/MongoDB, OpenCV, NumPy, InsightFace, pytest

## Global Constraints

- Challenge TTL is 60 seconds and every verification attempt consumes it once.
- Maximum video is 10 MB and 5 seconds; minimum is 12 sampled valid frames.
- Support WebM and MP4 without persisting uploaded media.
- Final score is `min(passive_score, motion_score)`; thresholds are server-owned.

---

### Task 1: Challenge contracts and atomic storage

**Files:** `app/config.py`, `app/schemas/face.py`, `app/database.py`, `tests/mongo_fakes.py`, `tests/test_video_liveness_api.py`

- [x] Write failing tests for authenticated issue, expiry, and one-time atomic consumption.
- [x] Add schemas/settings, Mongo indexes, fake atomic update support, and challenge router helpers.
- [x] Run focused tests and commit-ready verification.

### Task 2: Video and motion analysis

**Files:** `app/services/video_liveness.py`, `tests/test_video_liveness.py`

- [x] Write failing tests for turn/blink sequences, conservative passive score, composite score, face failures, and cleanup.
- [x] Implement bounded streaming decode, sampling, passive analysis, pose/EAR extraction, ordered motion state, and stable typed failures.
- [x] Run focused tests and commit-ready verification.

### Task 3: Verify-video endpoint and delivery

**Files:** `app/routers/face.py`, `tests/test_video_liveness_api.py`, `.env.example`, `README.md`

- [x] Write failing endpoint response/reason-code tests.
- [x] Integrate atomic challenge consumption with `VideoLivenessService`, multipart size/type validation, logging, and stable responses.
- [x] Document API/config, run all tests and diff checks.
- [ ] Commit and push to `origin/develop`.
