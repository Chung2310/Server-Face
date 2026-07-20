# ERP InsightFace Attendance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require GPS, passive liveness, and registered-face verification for ERP check-in/out, with permissioned enrollment and private Cloudinary evidence/audit history.

**Architecture:** The Express backend is the sole orchestrator. React captures camera/GPS and submits multipart data to ERP; ERP validates identity/tenant, uploads private evidence, checks GPS, calls InsightFace with server-only credentials, records an immutable attempt, and conditionally mutates attendance.

**Tech Stack:** TypeScript 5.8, Express 4, React 19, MongoDB/Mongoose, Joi, Multer, Cloudinary, Vitest.

## Global Constraints

- Browser code never receives InsightFace URL/key or Cloudinary signing credentials.
- Authenticated MongoDB user `_id` is the InsightFace `user_id`.
- Only full GPS + registration + face + liveness success mutates attendance.
- Failed evidence expires after 30 days; metadata remains.
- `face:manage` is default for admin/superadmin and assignable to HR/custom roles.
- All employee targets are tenant and hierarchy checked.

---

### Task 1: Environment and typed InsightFace client

**Files:**
- Modify: `.env.example`
- Modify: `server/config/env.ts`
- Create: `server/service/insightface.service.ts`
- Create: `server/service/insightface.service.test.ts`

**Interfaces:**
- Produces `verifyEmployee(userId: string, image: Buffer, mimeType: string): Promise<FaceVerificationResult>`.
- Produces `getRegistrationStatus`, `registerFace`, and `deleteRegistration` with typed results.
- Maps upstream timeout/unavailable/malformed response to `InsightFaceUnavailableError`; maps business reason codes without trusting prose.

- [ ] **Step 1: Write failing header, multipart, timeout, malformed-response, and env tests**
- [ ] **Step 2: Run `npx vitest run server/service/insightface.service.test.ts` and verify failure**
- [ ] **Step 3: Implement server-only env access and `fetch` client with `AbortSignal.timeout`**
- [ ] **Step 4: Run focused tests and verify PASS**

### Task 2: Private Cloudinary evidence primitives

**Files:**
- Modify: `server/service/cloudinary.service.ts`
- Create: `server/service/cloudinary.service.test.ts`

**Interfaces:**
- Produces `uploadPrivateImage(buffer, folder): Promise<{publicId, resourceType, type, format, bytes}>`.
- Produces `createSignedImageUrl(publicId, expiresAt): string` and `deleteAsset(publicId): Promise<void>`.

- [ ] **Step 1: Write failing upload-option, signed-URL, and deletion tests**
- [ ] **Step 2: Implement private/authenticated upload and identifier-preserving result**
- [ ] **Step 3: Run focused tests and verify PASS**

### Task 3: Permission catalog and enrollment audit persistence

**Files:**
- Modify: `server/middleware/auth.ts`
- Modify or seed through: `server/service/permission.service.ts`
- Modify: `src/components/user-admin/RoleModal.tsx`
- Create: `server/model/face-enrollment-audit.model.ts`
- Create: `server/interface/face-enrollment.interface.ts`
- Create: `server/controller/face-management.controller.ts`
- Create: `server/router/face-management.router.ts`
- Modify: `server/router/index.ts`
- Create: `server/controller/face-management.controller.test.ts`

**Interfaces:**
- Permission code: `face:manage`.
- Routes: status/register/delete under `/api/v1/face-management/users/:id`, all requiring auth, permission, company, and hierarchy checks.
- Multipart register accepts one image, uploads private evidence, calls InsightFace, and audits actor/target/result.

- [ ] **Step 1: Write failing admin-default, ordinary-denied, HR-granted, and cross-tenant tests**
- [ ] **Step 2: Add permission and secured routes with multipart limits**
- [ ] **Step 3: Add immutable enrollment audit and compensating Cloudinary cleanup**
- [ ] **Step 4: Run focused tests and verify PASS**

### Task 4: Attendance-attempt model and shared orchestrator

**Files:**
- Create: `server/interface/attendance-attempt.interface.ts`
- Create: `server/model/attendance-attempt.model.ts`
- Create: `server/service/timekeeping-attempt.service.ts`
- Create: `server/service/timekeeping-attempt.service.test.ts`
- Modify: `server/model/timekeeping.model.ts`
- Modify: `server/interface/timekeeping.interface.ts`

**Interfaces:**
- `executeAttendanceAttempt({action, uid, companyCode, latitude, longitude, deviceInfo, ipAddress, image, mimeType})` returns normalized success/rejection.
- Immutable attempt stores evidence identifiers and `evidenceDeleteAfter = attemptedAt + 30 days` only when failed.
- Successful timekeeping detail references `attemptId` and evidence `publicId`.

- [ ] **Step 1: Write failing matrix tests for GPS, registration, face, liveness, upstream error, and duplicate mutation**
- [ ] **Step 2: Implement schema/indexes and Haversine helper extraction**
- [ ] **Step 3: Implement upload, verification, immutable attempt, and conditional attendance mutation**
- [ ] **Step 4: Prove concurrent calls create one check-in/out and run focused tests**

### Task 5: Multipart check-in/out routes

**Files:**
- Modify: `server/router/timekeeping.router.ts`
- Modify: `server/controller/timekeeping.controller.ts`
- Create: `server/controller/timekeeping.controller.test.ts`

**Interfaces:**
- Existing POST paths remain, but consume multipart `file`, `latitude`, `longitude`, `deviceInfo`.
- Responses contain stable `reasonCode`, user-safe message, and attendance data only on success.

- [ ] **Step 1: Write failing multipart/size/MIME/auth/business-error tests**
- [ ] **Step 2: Wire Multer memory upload and shared orchestrator**
- [ ] **Step 3: Run focused tests and verify PASS**

### Task 6: Failed-evidence retention and authorized delivery

**Files:**
- Create: `server/service/attendance-evidence.service.ts`
- Create: `server/service/attendance-evidence.service.test.ts`
- Create: `server/controller/attendance-evidence.controller.ts`
- Create: `server/router/attendance-evidence.router.ts`
- Modify: `server/router/index.ts`
- Modify: `server.ts`

**Interfaces:**
- Idempotent `deleteExpiredFailedEvidence(now: Date)` deletes Cloudinary first and then stamps `evidenceDeletedAt`.
- Evidence endpoint returns a short-lived signed URL only after self/tenant/permission checks.

- [ ] **Step 1: Write failing retention boundary, retry, self-access, manager, and cross-tenant tests**
- [ ] **Step 2: Implement retention service and authorized URL endpoint**
- [ ] **Step 3: Register a bounded scheduled job with overlap protection**
- [ ] **Step 4: Run focused tests and verify PASS**

### Task 7: Reusable camera capture and dashboard workflow

**Files:**
- Create: `src/components/timekeeping/CameraCaptureModal.tsx`
- Create: `src/components/timekeeping/CameraCaptureModal.test.tsx`
- Modify: `src/pages/DashboardTab.tsx`
- Create: `src/services/timekeepingService.ts`
- Create: `src/services/timekeepingService.test.ts`

**Interfaces:**
- Modal returns a correctly oriented JPEG `File` and supports retake/cancel.
- Service sends multipart without manually setting `Content-Type` and preserves JWT auth.

- [ ] **Step 1: Write failing permission, capture, retake, cleanup, and multipart tests**
- [ ] **Step 2: Implement camera lifecycle and client-side image limits**
- [ ] **Step 3: Replace direct GPS JSON submit with camera + GPS modal flow**
- [ ] **Step 4: Run focused UI tests and verify PASS**

### Task 8: Employee face-management UI and attempt history

**Files:**
- Create: `src/components/user-admin/FaceManagementModal.tsx`
- Create: `src/components/user-admin/FaceManagementModal.test.tsx`
- Modify the employee action surface selected during implementation under `src/components/user-admin/`
- Modify: `src/components/hr/CalendarTab.tsx`
- Create: `src/services/faceManagementService.ts`

**Interfaces:**
- UI visibility is permission-driven; status supports not-set/set/unavailable.
- History requests evidence URLs on demand and displays unavailable/deleted evidence safely.

- [ ] **Step 1: Write failing permission-visibility and status/action tests**
- [ ] **Step 2: Implement setup/replace/delete modal using reusable camera capture**
- [ ] **Step 3: Add attempt/evidence history with signed URLs on demand**
- [ ] **Step 4: Run focused UI tests and verify PASS**

### Task 9: Full verification and deployment documentation

**Files:**
- Modify: `README.md`
- Create: `docs/deployment/insightface-attendance.md`

- [ ] **Step 1: Document env, model provisioning, HTTPS, retention, rollout flag, and rollback**
- [ ] **Step 2: Run `npm run typecheck`**
Expected: exit 0.

- [ ] **Step 3: Run `npx vitest run`**
Expected: exit 0.

- [ ] **Step 4: Run `npm run build`**
Expected: exit 0.

