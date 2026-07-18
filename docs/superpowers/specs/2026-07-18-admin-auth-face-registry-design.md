# Admin Authentication and Face Registry API Design

## Goal

Add persistent administrator authentication backed by the existing MongoDB instance, use secure HttpOnly session cookies for the admin panel, and expose API-key-protected operations for creating, deleting, and checking face registrations by `user_id`.

The existing face verification endpoint remains compatible with the current check-in/check-out integration.

## Scope

This change includes:

- Bootstrap of the initial administrator from environment variables.
- MongoDB-backed administrator accounts and sessions.
- Admin login and logout with an HttpOnly cookie.
- API-key authentication for face registration management endpoints.
- Image-based face registration.
- Registration status lookup without exposing embeddings.
- Face registration deletion.
- MongoDB startup checks and indexes.
- Admin panel integration.
- New configuration variables documented in `.env.example`.

This change does not include:

- A UI for creating additional administrator accounts.
- Multiple API keys, key rotation, roles, or permissions.
- A new MongoDB container.
- Returning face embeddings to clients.
- Changing the public contract of the existing check-in/check-out verification endpoint.

## Configuration

MongoDB continues to use the existing variables:

- `MONGODB_URI`
- `MONGODB_USER`
- `MONGODB_PASSWORD`
- `MONGODB_AUTH_SOURCE`

The application adds and documents these variables in `.env.example`:

- `ADMIN_USERNAME`: username for the bootstrap administrator.
- `ADMIN_PASSWORD`: plaintext bootstrap secret read only for bootstrap/login comparison input; only its password hash is stored in MongoDB.
- `ADMIN_SESSION_TTL_SECONDS`: administrator session lifetime.
- `ADMIN_COOKIE_SECURE`: enables the cookie `Secure` flag in HTTPS deployments.
- `FACE_API_KEY`: permanent integration key accepted through `X-API-Key`.
- `MONGODB_SERVER_SELECTION_TIMEOUT_MS`: bounded MongoDB connection timeout.

Secrets must never be logged. Production startup must not silently use insecure default secrets.

## Data Model

### `admins`

- `username`: unique administrator username.
- `password_hash`: password hash generated with a password-specific hashing algorithm.
- `is_active`: whether login is allowed.
- `created_at`: UTC creation timestamp.
- `updated_at`: UTC update timestamp.

At startup, the application checks for `ADMIN_USERNAME`. If it does not exist, it creates the account from `ADMIN_PASSWORD`. Existing administrator records are never overwritten during startup.

### `admin_sessions`

- `token_hash`: unique hash of a cryptographically random session token.
- `username`: administrator owning the session.
- `created_at`: UTC creation timestamp.
- `expires_at`: UTC expiry timestamp and MongoDB TTL index target.

Only the raw session token is placed in the browser cookie. MongoDB stores its hash.

### `face_registry`

- `user_id`: unique employee identifier.
- `embedding`: 512-dimensional face embedding.
- `created_at`: UTC time of initial registration.
- `updated_at`: UTC time of the latest registration.

Re-registering an existing `user_id` updates the embedding and `updated_at` while preserving `created_at`.

## Authentication

### Admin panel

`POST /api/v1/admin/login` accepts a username and password. On success it creates a database session and sets a cookie with:

- `HttpOnly=true`
- `SameSite=Lax`
- `Secure` controlled by `ADMIN_COOKIE_SECURE`
- bounded `Max-Age` derived from `ADMIN_SESSION_TTL_SECONDS`

`POST /api/v1/admin/logout` deletes the database session and clears the cookie. Admin dashboard endpoints validate the cookie and reject missing, expired, inactive, or unknown sessions with HTTP 401.

### Integration API

Face registry management endpoints require `X-API-Key`. The supplied value is compared with `FACE_API_KEY` using a timing-safe comparison. The API key is permanent until the environment variable is changed and the service is restarted.

The key is not embedded in the admin frontend. The browser uses its admin session cookie for admin-only equivalents needed by the interface.

## API Contracts

### Create or update a face registration

`POST /api/v1/face/register`

- Authentication: `X-API-Key`.
- Content type: `multipart/form-data`.
- Fields: `user_id`, `file`.
- Requires a decodable image containing exactly one face.
- Extracts the face embedding and upserts `face_registry`.
- Returns registration metadata without the embedding.

### Check registration readiness

`GET /api/v1/face/register/{user_id}`

- Authentication: `X-API-Key`.
- Returns HTTP 200 for both registered and unregistered identifiers.
- Registered response contains `user_id`, `registered: true`, `created_at`, and `updated_at`.
- Unregistered response contains `user_id`, `registered: false`, with null timestamps.

The endpoint answers only whether the employee is set up for check-in/check-out. It never returns the embedding.

### Delete a face registration

`DELETE /api/v1/face/register/{user_id}`

- Authentication: `X-API-Key`.
- Returns `user_id` and `deleted: true` when a record was deleted.
- Returns `user_id` and `deleted: false` when no record existed.

### Existing verification

`POST /api/v1/face/verify-employee` remains available with its existing `user_id` and image contract so current check-in/check-out clients are not disrupted.

## Admin Panel

The login form calls `/api/v1/admin/login` with credentials and relies on the HttpOnly cookie; it no longer builds or stores an HTTP Basic Authorization header. Logout calls `/api/v1/admin/logout`.

Admin metrics and face registry UI operations use the session cookie. The backend provides session-protected admin endpoints required to list, create, inspect, and delete registrations. Responses shown in the panel exclude embeddings.

## MongoDB Lifecycle

Application startup performs a bounded MongoDB ping, creates these indexes, and then bootstraps the configured administrator:

- Unique index on `admins.username`.
- Unique index on `admin_sessions.token_hash`.
- TTL index on `admin_sessions.expires_at`.
- Unique index on `face_registry.user_id`.

The project connects to the MongoDB container already running on the shared internal Docker network. `docker-compose.yml` does not add a MongoDB service.

If MongoDB is unavailable, health and admin metrics expose a disconnected state. Endpoints requiring persistent data fail promptly with HTTP 503 rather than waiting for the driver default timeout.

## Error Handling

- Invalid or missing admin session: HTTP 401.
- Invalid or missing `X-API-Key`: HTTP 401.
- Database unavailable: HTTP 503.
- Undecodable image: HTTP 400.
- No face in registration image: HTTP 422.
- More than one face in registration image: HTTP 422.
- Invalid request fields: HTTP 422.

Error responses do not expose credentials, connection strings, embeddings, or internal stack traces.

## Testing

Automated tests cover:

- Initial admin bootstrap and idempotent restart behavior.
- Password hashing and rejection of incorrect credentials.
- Login cookie flags, authenticated access, logout, and session expiry.
- Missing, incorrect, and correct API keys.
- Image-based create/update, status lookup, and deletion.
- Preservation of `created_at` during re-registration.
- Invalid images, no-face images, and multi-face images.
- Prompt HTTP 503 responses when MongoDB is unavailable.
- Existing face verification behavior.
- Admin UI endpoint contracts.

The final verification runs the full automated suite and an integration test against the configured MongoDB instance when it is reachable from the project environment.
