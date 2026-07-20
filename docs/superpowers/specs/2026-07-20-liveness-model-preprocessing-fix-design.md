# Liveness Model Preprocessing Fix Design

## Goal

Make passive liveness inference use the input distribution expected by the deployed 80x80 ONNX model so genuine camera captures are not systematically classified as spoofed.

## Root Cause

The current pipeline divides BGR pixels by 255 and uses a 1.2 face crop scale. Controlled inference on the reported image produced a live-class probability of 0.00549. Keeping float32 pixels in the 0-255 range and using a 2.7 crop scale produced 0.99923 on the same model and detected face.

## Design

- Resize the expanded BGR face crop to the configured input size.
- Cast pixels to `float32` without dividing by 255.
- Change the model-specific crop scale from 1.2 to 2.7.
- Preserve NCHW layout, contiguous memory, CPU inference, live class index 1, and threshold 0.8.
- Do not weaken liveness enforcement or change public API responses.

## Testing

Update the focused preprocessing test to assert the 2.7 crop geometry and unscaled float32 pixel values. Keep threshold-boundary, malformed-output, missing-model, health, registration, and secure-verification coverage passing.

## Delivery

Run the focused liveness tests and the complete test suite, commit the fix, and push it to `origin/develop`.
