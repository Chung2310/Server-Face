from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class FaceLandmark(BaseModel):
    x: float
    y: float

class FaceDetectionResponse(BaseModel):
    bbox: BoundingBox
    confidence: float = Field(..., description="Detection confidence score")
    landmarks: List[FaceLandmark] = Field(..., description="5 facial landmarks (eyes, nose, mouth corners)")
    gender: Optional[str] = None
    age: Optional[float] = None

class FaceEmbeddingResponse(BaseModel):
    bbox: BoundingBox
    embedding: List[float] = Field(..., description="512-dimensional face embedding")

class VerifyEmbeddingsRequest(BaseModel):
    embedding1: List[float] = Field(..., description="First 512-dimensional embedding")
    embedding2: List[float] = Field(..., description="Second 512-dimensional embedding")

class VerifyResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity score")
    verified: bool = Field(..., description="True if similarity meets or exceeds threshold")
    threshold: float = Field(..., description="The similarity threshold used")

class SearchRegisterRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    embedding: List[float] = Field(..., description="512-dimensional embedding to register")

class SearchQueryRequest(BaseModel):
    embedding: List[float] = Field(..., description="The query face embedding to search for")
    limit: int = Field(5, description="Maximum number of matches to return")

class SearchMatch(BaseModel):
    user_id: str
    similarity: float

class SearchResponse(BaseModel):
    matches: List[SearchMatch]

class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class AdminMeResponse(BaseModel):
    username: str

class FaceRegistrationInfo(BaseModel):
    user_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class RegisterFaceResponse(BaseModel):
    user_id: str
    registered: bool = True
    created: bool = Field(..., description="True if this was a new registration")
    created_at: datetime
    updated_at: datetime

class RegistrationStatusResponse(BaseModel):
    user_id: str
    registered: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DeleteRegistrationResponse(BaseModel):
    user_id: str
    deleted: bool

class VerifyEmployeeResponse(BaseModel):
    verified: bool = Field(..., description="True if verification succeeds")
    reason: str = Field(..., description="Vietnamese explanation of the verification result")
    similarity: Optional[float] = Field(None, description="Similarity score between the faces")
class SecureVerifyEmployeeResponse(BaseModel):
    registered: bool
    face_verified: bool
    similarity: Optional[float] = None
    face_threshold: float
    live: bool
    liveness_score: Optional[float] = None
    liveness_threshold: float
    reason_code: str

class VideoChallengeResponse(BaseModel):
    challenge_id: str = Field(..., description="Opaque one-time challenge identifier")
    action: str = Field(..., description="Requested motion action: turn_left, turn_right, or blink")
    expires_in_seconds: int = Field(..., description="Seconds until the challenge expires unused")

class VideoLivenessVerifyResponse(BaseModel):
    verified: bool
    liveness_score: float
    passive_score: float
    motion_score: float
    reason_code: str
    sampled_frames: int
