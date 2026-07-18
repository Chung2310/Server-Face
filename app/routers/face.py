import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from app.services.face_analysis import FaceAnalysisService
from app.schemas.face import (
    FaceDetectionResponse, BoundingBox, FaceLandmark,
    FaceEmbeddingResponse, VerifyEmbeddingsRequest, VerifyResponse,
    SearchRegisterRequest, SearchQueryRequest, SearchResponse, SearchMatch,
    VerifyEmployeeResponse
)
from app.config import settings

from datetime import datetime
from app.database import db

router = APIRouter(prefix="/face", tags=["Face Operations"])
service = FaceAnalysisService()

def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    return img

def format_detection_response(face) -> FaceDetectionResponse:
    # face.bbox is [x1, y1, x2, y2]
    bbox = BoundingBox(
        x1=float(face.bbox[0]),
        y1=float(face.bbox[1]),
        x2=float(face.bbox[2]),
        y2=float(face.bbox[3])
    )
    # face.kps is 5x2 array
    landmarks = [FaceLandmark(x=float(kp[0]), y=float(kp[1])) for kp in face.kps]
    
    # gender: face.gender is 0 (F) or 1 (M) or vice-versa, or string in some versions
    gender_str = "M" if getattr(face, "gender", 0) == 1 else "F"
    
    return FaceDetectionResponse(
        bbox=bbox,
        confidence=float(face.det_score),
        landmarks=landmarks,
        gender=gender_str,
        age=float(getattr(face, "age", 0.0))
    )

@router.post("/detect", response_model=List[FaceDetectionResponse])
async def detect_faces(file: UploadFile = File(...)):
    """
    Detects all faces in the uploaded image, returning their bounding boxes, landmarks, gender, and age.
    """
    contents = await file.read()
    img = decode_image(contents)
    faces = service.detect_faces(img)
    return [format_detection_response(face) for face in faces]

@router.post("/embedding", response_model=List[FaceEmbeddingResponse])
async def extract_embeddings(
    file: UploadFile = File(...),
    largest_only: bool = Query(True, description="If True, only return the embedding for the largest face detected")
):
    """
    Extracts the 512-dimensional face embedding for face(s) in the uploaded image.
    """
    contents = await file.read()
    img = decode_image(contents)
    faces = service.detect_faces(img)
    
    if not faces:
        raise HTTPException(status_code=404, detail="No faces detected in the image")
        
    if largest_only:
        # Sort and pick the largest face by bounding box area
        largest_face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
        faces_to_process = [largest_face]
    else:
        faces_to_process = faces

    response = []
    for face in faces_to_process:
        bbox = BoundingBox(
            x1=float(face.bbox[0]),
            y1=float(face.bbox[1]),
            x2=float(face.bbox[2]),
            y2=float(face.bbox[3])
        )
        response.append(FaceEmbeddingResponse(
            bbox=bbox,
            embedding=face.embedding.tolist()
        ))
    return response

@router.post("/verify-images", response_model=VerifyResponse)
async def verify_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Detects the largest face in each of the two uploaded images and compares them.
    """
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    img1 = decode_image(contents1)
    img2 = decode_image(contents2)
    
    face1 = service.get_largest_face(img1)
    face2 = service.get_largest_face(img2)
    
    if not face1:
        raise HTTPException(status_code=400, detail="No face detected in Image 1")
    if not face2:
        raise HTTPException(status_code=400, detail="No face detected in Image 2")
        
    similarity = service.compute_similarity(face1.embedding, face2.embedding)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold
    
    return VerifyResponse(
        similarity=similarity,
        verified=verified,
        threshold=active_threshold
    )

@router.post("/verify-embeddings", response_model=VerifyResponse)
async def verify_embeddings(
    request: VerifyEmbeddingsRequest,
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Compares two pre-extracted 512-dimensional face embeddings.
    """
    emb1 = np.array(request.embedding1, dtype=np.float32)
    emb2 = np.array(request.embedding2, dtype=np.float32)
    
    if emb1.shape != (512,) or emb2.shape != (512,):
        raise HTTPException(status_code=400, detail="Embeddings must be 512-dimensional vectors")
        
    similarity = service.compute_similarity(emb1, emb2)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold
    
    return VerifyResponse(
        similarity=similarity,
        verified=verified,
        threshold=active_threshold
    )

@router.post("/register", response_model=dict)
async def register_face(request: SearchRegisterRequest):
    """
    Registers a user ID and their corresponding face embedding in MongoDB.
    """
    emb = np.array(request.embedding, dtype=np.float32)
    if emb.shape != (512,):
        raise HTTPException(status_code=400, detail="Embedding must be a 512-dimensional vector")
        
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    # Upsert the face embedding in MongoDB face_registry collection
    await db.face_registry.update_one(
        {"user_id": request.user_id},
        {
            "$set": {
                "user_id": request.user_id,
                "embedding": request.embedding,
                "updated_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    return {"message": f"Successfully registered user: {request.user_id}"}

@router.post("/search", response_model=SearchResponse)
async def search_face(
    request: SearchQueryRequest,
    threshold: float = Query(None, description="Optional custom similarity threshold override")
):
    """
    Searches the MongoDB registry for matching faces using the query embedding.
    """
    query_emb = np.array(request.embedding, dtype=np.float32)
    if query_emb.shape != (512,):
        raise HTTPException(status_code=400, detail="Query embedding must be a 512-dimensional vector")
        
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    matches = []
    
    # Query all records from MongoDB face_registry
    cursor = db.face_registry.find({}, {"user_id": 1, "embedding": 1})
    async for doc in cursor:
        reg_emb = np.array(doc["embedding"], dtype=np.float32)
        sim = service.compute_similarity(query_emb, reg_emb)
        if sim >= active_threshold:
            matches.append(SearchMatch(user_id=doc["user_id"], similarity=sim))
            
    # Sort matches by similarity descending
    matches.sort(key=lambda x: x.similarity, reverse=True)
    
    return SearchResponse(matches=matches[:request.limit])

@router.post("/verify-employee", response_model=VerifyEmployeeResponse)
async def verify_employee(
    user_id: str = Form(..., description="Mã nhân viên cần xác thực"),
    file: UploadFile = File(..., description="Ảnh chụp khuôn mặt để đối sánh"),
    threshold: float = Query(None, description="Ngưỡng xác thực tùy chỉnh (Mặc định: 0.45)")
):
    """
    Xác thực khuôn mặt nhân viên từ ảnh tải lên so với dữ liệu đã lưu trong MongoDB.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")

    # 1. Tìm kiếm thông tin đăng ký của nhân viên trong MongoDB
    doc = await db.face_registry.find_one({"user_id": user_id})
    if not doc:
        return VerifyEmployeeResponse(
            verified=False,
            reason="Nhân viên chưa đăng ký thông tin khuôn mặt trên hệ thống.",
            similarity=None
        )

    registered_emb = np.array(doc["embedding"], dtype=np.float32)

    # 2. Giải mã file ảnh tải lên
    contents = await file.read()
    try:
        img = decode_image(contents)
    except HTTPException as e:
        return VerifyEmployeeResponse(
            verified=False,
            reason="File ảnh không hợp lệ hoặc bị lỗi định dạng.",
            similarity=None
        )

    # 3. Phát hiện khuôn mặt lớn nhất trong ảnh chụp mới
    face = service.get_largest_face(img)
    if not face:
        return VerifyEmployeeResponse(
            verified=False,
            reason="Không phát hiện thấy khuôn mặt nào trong ảnh chụp.",
            similarity=None
        )

    # 4. Tính toán độ tương đồng giữa khuôn mặt chụp và khuôn mặt đăng ký
    similarity = service.compute_similarity(registered_emb, face.embedding)
    active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    verified = similarity >= active_threshold

    # 5. Trả về kết quả
    if verified:
        reason = "Xác thực khuôn mặt thành công."
    else:
        reason = f"Xác thực thất bại. Khuôn mặt không khớp với nhân viên đã đăng ký (Độ khớp: {similarity*100:.1f}% < {active_threshold*100:.1f}%)."

    return VerifyEmployeeResponse(
        verified=verified,
        reason=reason,
        similarity=similarity
    )
