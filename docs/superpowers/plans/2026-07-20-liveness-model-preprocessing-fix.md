# Liveness Model Preprocessing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the deployed 80x80 liveness ONNX model the crop geometry and pixel range it expects.

**Architecture:** Keep preprocessing inside `LivenessService` and make the smallest model-specific correction: expand the detected face by 2.7 and cast resized BGR pixels to contiguous NCHW float32 without normalization. Preserve inference, scoring, thresholds, and API behavior.

**Tech Stack:** Python 3.10, NumPy, OpenCV, ONNX Runtime, pytest

## Global Constraints

- Keep `LIVENESS_LIVE_CLASS_INDEX=1` and `LIVENESS_THRESHOLD=0.8`.
- Do not weaken liveness enforcement or change public API responses.
- Push the verified implementation to `origin/develop`.

---

### Task 1: Correct liveness crop and pixel range

**Files:**
- Modify: `tests/test_liveness.py`
- Modify: `app/services/liveness.py:21`
- Modify: `app/services/liveness.py:108-113`

**Interfaces:**
- Consumes: BGR `np.ndarray` images and face bounding boxes.
- Produces: contiguous tensors with shape `(1, 3, height, width)`, dtype `float32`, and pixel range `0-255`.

- [ ] **Step 1: Write failing preprocessing tests**

Change the expected tensor to omit `/ 255.0`. Add a crop geometry test using a 100x100 image and `[40, 40, 60, 60]` face box; monkeypatch `cv2.resize` and assert the source crop is 54x54, proving scale 2.7.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_liveness.py -q -p no:cacheprovider --basetemp D:\tmp\liveness-red
```

Expected: tensor range assertion and crop size assertion fail against `/255` and scale 1.2.

- [ ] **Step 3: Implement the minimal preprocessing correction**

Set `_crop_scale = 2.7` and replace normalization with:

```python
tensor = resized.astype(np.float32)
return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
```

- [ ] **Step 4: Run focused and full verification**

Run the focused command from Step 2, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp D:\tmp\liveness-full-green
```

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```powershell
git add app/services/liveness.py tests/test_liveness.py docs/superpowers/plans/2026-07-20-liveness-model-preprocessing-fix.md
git commit -m "fix: correct liveness model preprocessing"
git push origin develop
```
