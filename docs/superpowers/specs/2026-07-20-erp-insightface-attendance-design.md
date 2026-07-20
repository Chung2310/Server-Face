# ERP InsightFace Attendance Integration Design

## Goal

Integrate the InsightFace service with Igen ERP so a check-in or check-out is accepted only when the authenticated employee is inside the configured GPS radius, matches their registered face, and passes passive liveness detection. Provide a separately authorized face-setup workflow for administrators and roles such as HR.

## Scope

This delivery spans both sibling repositories:

- `Igen-ERP`: browser camera experience, GPS collection, server-side orchestration, permissions, Cloudinary evidence, attendance attempts, retention, and attendance history.
- `InsightFace`: API-key-protected registration status and face verification with CPU-first passive liveness.

Active challenge liveness, native mobile applications, multiple enrollment photos, and migration to a commercial presentation-attack-detection provider are outside this phase.

## Architecture

Igen ERP is the only orchestrator and source of truth for identity, tenant membership, permissions, GPS rules, attendance state, evidence retention, and audit history. The browser calls only ERP endpoints with the existing JWT. It never receives the InsightFace base URL or API key.

ERP calls InsightFace server-to-server using `INSIGHTFACE_BASE_URL` and `INSIGHTFACE_API_KEY` from the ERP environment. InsightFace is responsible only for registration metadata, embedding persistence, face matching, and passive liveness inference. Integration endpoints require `X-API-Key`.

Cloudinary stores private attendance and enrollment evidence. ERP stores Cloudinary's asset identifier as well as delivery metadata so it can generate authorized view URLs and delete expired assets. A successful attendance mutation and its attempt record are coordinated by ERP; InsightFace never writes ERP attendance data.

## Configuration

Igen ERP adds these server-only environment variables:

- `INSIGHTFACE_BASE_URL`: origin of the InsightFace service, without a trailing slash.
- `INSIGHTFACE_API_KEY`: value sent in `X-API-Key`.
- `INSIGHTFACE_TIMEOUT_MS`: request timeout, default `5000`.
- `ATTENDANCE_FAILED_EVIDENCE_RETENTION_DAYS`: default `30`.

InsightFace continues to use `FACE_API_KEY` as the expected `X-API-Key`. It adds explicit liveness model path, threshold, and CPU provider settings. Production startup fails if an integration key or required liveness model is missing; verification must not silently run without liveness.

Secrets are documented with placeholders in `.env.example`. Real values remain only in untracked `.env` or deployment secret storage.

## InsightFace Contract

### Registration status

`GET /api/v1/face/register/{user_id}` remains API-key protected and returns metadata only. ERP uses the authenticated ERP MongoDB user `_id` as `user_id`.

### Register or replace a face

`POST /api/v1/face/register` accepts multipart `user_id` and one image. It requires exactly one detectable, sufficiently large, front-facing face and a passing liveness result. On success it upserts the embedding and returns timestamps. Embeddings are never returned.

Enrollment failure categories are stable machine codes rather than Vietnamese prose only: `invalid_image`, `no_face`, `multiple_faces`, `face_too_small`, `poor_pose`, `spoof_suspected`, and `model_unavailable`.

### Verify an employee

The secured server-to-server verification contract accepts multipart `user_id` and `file`. It detects exactly one usable face, runs passive liveness, compares the embedding with the registered embedding, and returns:

```json
{
  "registered": true,
  "face_verified": true,
  "similarity": 0.71,
  "face_threshold": 0.45,
  "live": true,
  "liveness_score": 0.93,
  "liveness_threshold": 0.80,
  "reason_code": "verified"
}
```

Stable verification reason codes include `verified`, `not_registered`, `invalid_image`, `no_face`, `multiple_faces`, `face_too_small`, `spoof_suspected`, `face_mismatch`, and `model_unavailable`.

The existing public verification behavior is not trusted by ERP. The implementation either protects the existing `/verify-employee` route without known external consumers or introduces a secured `/verify-employee-secure` route and deprecates the public route. The implementation plan must preserve compatibility until repository usage and deployment consumers are verified.

## Passive Liveness

The first CPU candidate is a benchmarked ONNX MiniFASNet/Silent-Face-Anti-Spoofing model because it is designed as a lightweight presentation-attack detector. Selection is not based on model size alone. Before enabling production enforcement, the team builds a representative validation set covering company devices, lighting, skin tones, glasses, masks where permitted, printed photos, and phone/tablet replay attacks.

The threshold is chosen from measured false-accept and false-reject rates and is configuration, not a client-controlled query parameter. Both enrollment and attendance fail closed when the liveness model is unavailable. Health output exposes model readiness without exposing paths or secrets.

Passive single-image RGB liveness reduces common print/display attacks but does not guarantee protection against every replay, injection, deepfake, or unseen presentation attack. The UI must not claim absolute spoof prevention. A future phase may add randomized active challenges or a specialized vendor if benchmark results do not satisfy the business acceptance threshold.

## Permission Model and Face Setup

Add the permission code `face:manage` to ERP's permission catalog and Role Setup UI.

- `superadmin` and `admin` receive `face:manage` by default.
- Other roles, including a company-defined HR role, receive it only when an authorized role administrator assigns it.
- Ordinary employees cannot enroll themselves in this phase.
- Every setup/status/delete route requires authentication, `face:manage`, same-company object access, and existing hierarchy checks where applicable.

The face-management UI lives with employee management rather than global system settings. For an employee it displays `not set up`, `set up`, or `service unavailable`, the last enrollment timestamp, and authorized actions to set up, replace, or delete the registration.

Setup opens a dedicated camera modal. The operator grants camera permission, centers one employee, captures one front-facing image, previews it, and can retake before submitting. ERP uploads the private enrollment evidence to Cloudinary and then registers it with InsightFace. ERP records who performed the action, target employee, timestamps, result, Cloudinary asset identifier, and the active enrollment evidence reference. Replacing or deleting a face creates an audit event; superseded private enrollment assets follow an explicit cleanup path rather than becoming orphaned.

## Check-In and Check-Out Experience

The existing dashboard timekeeping buttons open a capture modal instead of immediately submitting GPS JSON.

1. The modal requests camera access and GPS access only after the user initiates check-in or check-out.
2. It shows a mirrored live preview for user comfort, but the captured image has the correct orientation.
3. The employee captures one image, reviews it, and may retake it.
4. Submit sends one multipart request to the corresponding ERP endpoint with `file`, `latitude`, `longitude`, and `deviceInfo`.
5. While processing, duplicate submission is disabled. Camera tracks stop on submit, cancel, navigation, logout, and component unmount.
6. The result distinguishes permission denial, outside radius, face not set up, spoof suspected, face mismatch, unusable photo, AI unavailable, upload failure, duplicate attendance, and success.

If the user is not registered, check-in/check-out is rejected with guidance to contact an administrator or a role holding `face:manage`. The employee is not redirected into self-enrollment.

The browser applies an image size and JPEG quality limit before upload, while ERP independently validates MIME type, decoded image type, and maximum bytes. Camera and geolocation require HTTPS outside localhost.

## ERP Attendance Orchestration

Both check-in and check-out share one focused orchestration service so validation order and audit behavior cannot drift.

1. Authenticate the JWT and derive `uid` and `companyCode`; ignore any client-supplied identity.
2. Validate multipart fields and enforce upload limits.
3. Load company location configuration and calculate Haversine distance server-side.
4. Upload the submitted image as a private Cloudinary asset under a tenant/user/date-scoped folder.
5. Call InsightFace with the authenticated ERP `uid` and image using a bounded timeout and `X-API-Key`.
6. Persist an immutable attendance-attempt record containing action, coordinates, distance, GPS result, registration result, face and liveness scores/results, reason code, device/IP metadata, evidence asset identifier, and timestamps.
7. Only when GPS, registration, face match, and liveness all pass does ERP atomically create check-in or update check-out under the existing daily uniqueness and sequencing rules.

GPS and biometric evaluation both run for a submitted attempt so the audit record contains both condition results, including attempts outside the allowed radius. A service timeout or malformed response is `verification_unavailable` and fails closed. Client-provided thresholds are never accepted.

Concurrent double submissions are handled by database constraints/conditional updates, not only UI disabling. If evidence upload succeeds but database persistence fails, a compensating cleanup job records and removes the orphaned Cloudinary asset.

## Data Model

Extend each successful `checkIn` and `checkOut` detail with a reference to its successful attempt and evidence metadata needed by history views. Do not copy all biometric fields into the daily log.

Create an immutable `AttendanceAttempt` collection with at least:

- `uid`, `companyCode`, `date`, `action` (`check_in` or `check_out`), and `attemptedAt`;
- submitted latitude/longitude, computed distance, allowed radius, and `gpsPassed`;
- `registered`, `faceVerified`, `similarity`, face threshold;
- `live`, `livenessScore`, liveness threshold;
- normalized `reasonCode`, overall `passed`, and upstream request correlation ID;
- Cloudinary `publicId`, resource type, delivery/access type, format, bytes, and authorized delivery reference;
- device info and IP address;
- `evidenceDeleteAfter` for failed attempts and `evidenceDeletedAt` after deletion.

Indexes support tenant/user/date history, recent failed-attempt review, and retention scans. An attempt document never stores an embedding or API key.

Create a face-enrollment audit model or focused audit subtype containing target employee, actor, company, action, result, evidence metadata, and timestamps. The active enrollment evidence reference is retrievable only through an authorized ERP endpoint that produces a short-lived signed Cloudinary URL.

## Evidence Access and Retention

Attendance and enrollment images are uploaded as private/authenticated Cloudinary assets. Raw public `secure_url` values are not exposed as permanent links. ERP verifies JWT, tenant scope, and the relevant attendance/face permission before returning a short-lived signed delivery URL.

- Successful attendance evidence is retained with attendance history according to the existing company record-retention policy.
- Failed-attempt evidence is deleted after 30 days.
- Failed-attempt metadata remains for audit after its image is deleted.
- Enrollment evidence is retained while active; replacement/deletion records an audit event and schedules the superseded asset for cleanup according to the same sensitive-evidence policy.

A scheduled idempotent retention job selects due assets, deletes them from Cloudinary by `publicId`, and only then stamps `evidenceDeletedAt`. Failures are retried and observable.

## Attendance History and Audit UI

Authorized attendance history shows the successful evidence linked to check-in/check-out. A separate attempts view shows failed and successful attempts with time, employee, action, GPS outcome/distance, face result, liveness result, normalized reason, and evidence availability.

Ordinary employees may view their own successful attendance evidence and their own attempts. Managers/HR/admin access follows existing tenant, hierarchy, and permission rules; `face:manage` alone does not grant unrestricted attendance-history access. Signed image URLs are generated only when the viewer opens evidence and expire quickly.

## Error and Security Behavior

- Missing/invalid ERP JWT: `401`.
- Missing `face:manage` for setup operations: `403`.
- Cross-company employee target: `403` or non-enumerating `404`, consistently with existing ERP conventions.
- Missing/invalid InsightFace API key: `401`; missing server configuration/model: `503`.
- Invalid/missing multipart fields: `400` or validation-standard `422`.
- Oversized/unsupported image: `413`/`415`.
- Outside GPS radius, spoof, mismatch, unregistered, duplicate, and invalid attendance sequence are business rejections with stable reason codes and user-safe Vietnamese messages.
- InsightFace timeout/unavailability: `503` from the orchestration layer with `verification_unavailable`; no attendance mutation.
- Cloudinary upload failure: no InsightFace request and no attendance mutation; persist operational failure metadata when possible without claiming image evidence exists.

Logs include correlation IDs and normalized reason codes but never API keys, embeddings, raw image bytes, or signed evidence URLs. Rate limits apply to verification and face-management endpoints per authenticated user, IP, and tenant. Server-side request construction prevents arbitrary InsightFace URLs and SSRF.

## Testing and Acceptance

### InsightFace

- API-key tests for status, register, delete, and secured verification.
- Liveness preprocessing, threshold-boundary, live/spoof, unavailable-model, and malformed-model-output tests.
- Registration/verification tests for no face, multiple faces, small face, mismatch, unregistered employee, and success.
- Contract tests assert stable reason codes and that embeddings are never returned.
- CPU benchmark records p50/p95 inference latency and memory usage with the production image limit.

### ERP server

- Environment validation and InsightFace client timeout/header/response-mapping tests.
- Permission tests for admin default, ordinary user denial, HR role granted `face:manage`, tenant isolation, and hierarchy enforcement.
- Multipart validation and Cloudinary private upload metadata tests.
- Orchestration matrix covers every combination of GPS, registration, face, liveness, upstream failure, upload failure, duplicate request, and check-out-before-check-in.
- Persistence tests prove every submitted image-backed attempt is audited, only full success mutates attendance, and concurrent submissions produce one mutation.
- Retention tests prove only failed evidence older than 30 days is deleted and retries are idempotent.
- Evidence-access tests cover self, permitted manager/HR/admin, cross-tenant denial, expired signed URLs, and deleted assets.

### ERP browser

- Camera/GPS permission denial, unsupported browser, capture/retake, mirrored preview/correct output, submit locking, cleanup of media tracks, and error-message mapping.
- Setup UI visibility and actions follow `face:manage`.
- Check-in/out is not submitted before capture and both coordinates and image are present in multipart payload.

### Production acceptance

- HTTPS camera and GPS work on the supported desktop and mobile browsers.
- A representative local dataset establishes the selected liveness threshold and documents false-accept/false-reject results before enforcement.
- Printed-photo and phone/tablet replay cases used in acceptance are rejected at the agreed threshold.
- End-to-end p95 from submit to response meets the CPU-first target of approximately two seconds under expected load; if the benchmark cannot meet it, capacity or model choice is revisited rather than weakening checks silently.
- Secrets are absent from browser bundles, API responses, logs, and git-tracked files.

## Rollout

Deploy the secured InsightFace contract and liveness readiness checks first. Deploy ERP configuration, client, models, permissions, setup UI, attempts/evidence, and check-in flow behind a company feature flag. Enroll pilot employees, benchmark real devices and attacks, then enable enforcement tenant by tenant. Rollback disables the feature flag and restores the previous UI only as an explicit operational decision; it must never silently accept attendance after a verification outage.

