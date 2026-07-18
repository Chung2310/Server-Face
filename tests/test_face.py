import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# In-memory storage mock representing MongoDB collection
MOCK_DB_STORE = []

class MockCursor:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.data):
            raise StopAsyncIteration
        val = self.data[self.index]
        self.index += 1
        return val

async def mock_update_one(filter_query, update_query, upsert=False):
    user_id = filter_query["user_id"]
    embedding = update_query["$set"]["embedding"]
    # Remove existing record if present to simulate replacement
    for idx, doc in enumerate(MOCK_DB_STORE):
        if doc["user_id"] == user_id:
            MOCK_DB_STORE[idx] = {"user_id": user_id, "embedding": embedding}
            return MagicMock()
    MOCK_DB_STORE.append({"user_id": user_id, "embedding": embedding})
    return MagicMock()

def mock_find(query, projection=None):
    return MockCursor(list(MOCK_DB_STORE))

async def mock_find_one(filter_query):
    user_id = filter_query["user_id"]
    for doc in MOCK_DB_STORE:
        if doc["user_id"] == user_id:
            return doc
    return None

# Construct Mock DB
mock_db = MagicMock()
mock_db.face_registry.update_one = mock_update_one
mock_db.face_registry.find_one = mock_find_one
mock_db.face_registry.find = mock_find

# Configure the service mock
mock_service_instance = MagicMock()
mock_service_instance._initialized = True

def mock_sim(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 0.0
    return float(dot_product / (norm_emb1 * norm_emb2))
    
mock_service_instance.compute_similarity = mock_sim

# Start global patchers at the module level (no with blocks to prevent premature teardown)
service_patcher = patch("app.services.face_analysis.FaceAnalysisService.__new__", return_value=mock_service_instance)
service_patcher.start()

db_patcher = patch("app.database.db", mock_db)
db_patcher.start()

decode_patcher = patch("app.routers.face.decode_image", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
decode_patcher.start()

# Import the app with mocks active
from app.main import app
client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "OK"
    assert json_data["model_name"] == "buffalo_s"

def test_verify_embeddings_valid():
    # Simple unit vectors for exact matches
    emb1 = [1.0] + [0.0] * 511
    emb2 = [1.0] + [0.0] * 511
    
    payload = {
        "embedding1": emb1,
        "embedding2": emb2
    }
    response = client.post("/api/v1/face/verify-embeddings", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is True
    assert json_data["similarity"] == pytest.approx(1.0)

def test_verify_embeddings_dissimilar():
    emb1 = [1.0] + [0.0] * 511
    emb2 = [0.0, 1.0] + [0.0] * 510  # Orthogonal vector, similarity should be 0.0
    
    payload = {
        "embedding1": emb1,
        "embedding2": emb2
    }
    response = client.post("/api/v1/face/verify-embeddings", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is False
    assert json_data["similarity"] == pytest.approx(0.0)

def test_register_and_search():
    emb1 = [1.0] + [0.0] * 511
    emb2 = [0.0, 1.0] + [0.0] * 510

    # Register user1
    reg_response = client.post("/api/v1/face/register", json={
        "user_id": "user1",
        "embedding": emb1
    })
    assert reg_response.status_code == 200
    assert reg_response.json()["message"] == "Successfully registered user: user1"

    # Search for user1 (query with emb1)
    search_response = client.post("/api/v1/face/search", json={
        "embedding": emb1,
        "limit": 5
    })
    assert search_response.status_code == 200
    results = search_response.json()["matches"]
    assert len(results) > 0
    assert results[0]["user_id"] == "user1"
    assert results[0]["similarity"] == pytest.approx(1.0)

    # Search query with orthogonal emb2 - should not match user1 (threshold is 0.45, similarity is 0.0)
    search_response_empty = client.post("/api/v1/face/search", json={
        "embedding": emb2,
        "limit": 5
    })
    assert search_response_empty.status_code == 200
    assert len(search_response_empty.json()["matches"]) == 0

def test_verify_employee_not_registered():
    # Make request for unregistered user
    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "unregistered_user"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is False
    assert "chưa đăng ký" in json_data["reason"]

def test_verify_employee_success():
    emb1 = [1.0] + [0.0] * 511
    
    # 1. Pre-register the user
    client.post("/api/v1/face/register", json={
        "user_id": "user_verified_ok",
        "embedding": emb1
    })

    # 2. Mock service to detect face and return emb1
    mock_face = MagicMock()
    mock_face.embedding = np.array(emb1, dtype=np.float32)
    mock_service_instance.get_largest_face.return_value = mock_face

    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "user_verified_ok"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is True
    assert "thành công" in json_data["reason"]
    assert json_data["similarity"] == pytest.approx(1.0)

def test_verify_employee_fail():
    emb_registered = [1.0] + [0.0] * 511
    emb_detected = [0.0, 1.0] + [0.0] * 510  # Orthogonal vector
    
    # 1. Pre-register the user
    client.post("/api/v1/face/register", json={
        "user_id": "user_verified_fail",
        "embedding": emb_registered
    })

    # 2. Mock service to return orthogonal face embedding
    mock_face = MagicMock()
    mock_face.embedding = np.array(emb_detected, dtype=np.float32)
    mock_service_instance.get_largest_face.return_value = mock_face

    response = client.post(
        "/api/v1/face/verify-employee",
        data={"user_id": "user_verified_fail"},
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["verified"] is False
    assert "thất bại" in json_data["reason"]
    assert json_data["similarity"] == pytest.approx(0.0)

def test_admin_metrics_unauthorized():
    """Truy cập /admin/metrics không có xác thực phải trả về 401."""
    response = client.get("/api/v1/admin/metrics")
    assert response.status_code == 401

def test_admin_metrics_wrong_credentials():
    """Truy cập /admin/metrics với sai mật khẩu phải trả về 401."""
    import base64
    bad_creds = base64.b64encode(b"admin:wrongpassword").decode()
    response = client.get(
        "/api/v1/admin/metrics",
        headers={"Authorization": f"Basic {bad_creds}"}
    )
    assert response.status_code == 401

def test_admin_metrics_authorized():
    """Truy cập /admin/metrics với đúng tài khoản phải trả về 200 và các trường hợp lệ."""
    import base64
    from app.config import settings
    creds = base64.b64encode(
        f"{settings.ADMIN_USERNAME}:{settings.ADMIN_PASSWORD}".encode()
    ).decode()
    response = client.get(
        "/api/v1/admin/metrics",
        headers={"Authorization": f"Basic {creds}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "database" in data
    assert "service" in data
    assert "cpu_percent" in data["system"]
    assert "memory_percent" in data["system"]

# Clean up patchers at exit
@pytest.fixture(scope="session", autouse=True)
def cleanup_patchers():
    yield
    service_patcher.stop()
    db_patcher.stop()
    decode_patcher.stop()
