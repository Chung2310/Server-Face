# Detailed Liveness Error Logging Design

## Goal

Make `model_unavailable` failures diagnosable from server logs while preserving the existing stable and non-sensitive API response.

## Scope

- Update both liveness exception boundaries in `app/routers/face.py`: face registration and secure employee verification.
- Log the operation, `user_id`, exception message, and full traceback with `logger.exception`.
- Keep the HTTP status `503` and response body `{"detail":{"reason_code":"model_unavailable"}}` unchanged.
- Do not expose model paths, stack traces, or internal exception messages to API clients.

## Error Flow

When `LivenessService` raises `LivenessUnavailableError`, the router records the exception while the exception context is active. It then raises the same stable `HTTPException` already used by clients.

## Testing

Add focused regression tests for registration and secure verification. Each test causes liveness analysis to raise `LivenessUnavailableError`, asserts the existing 503 response, and verifies that an exception-level log contains the operation and affected `user_id`.

## Delivery

Run the focused tests and the full test suite, commit the implementation, and push the resulting commit to the remote `develop` branch.
