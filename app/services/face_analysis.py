import logging
import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from app.config import settings

logger = logging.getLogger("uvicorn.error")

class FaceAnalysisService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceAnalysisService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Determine ONNX Runtime provider dynamically
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")
        
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
            logger.info("Using GPU (CUDAExecutionProvider) for InsightFace")
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            logger.info("Using CPU (CPUExecutionProvider) for InsightFace")

        try:
            self.model = FaceAnalysis(name=settings.MODEL_NAME, providers=providers)
            self.model.prepare(ctx_id=ctx_id, det_size=settings.DET_SIZE)
            self._initialized = True
            logger.info(f"InsightFace model '{settings.MODEL_NAME}' successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            raise e

    def detect_faces(self, img: np.ndarray):
        """
        Detects faces in a BGR image.
        Returns a list of face objects.
        """
        return self.model.get(img)

    def get_largest_face(self, img: np.ndarray):
        """
        Helper to return only the largest face detected in the image.
        """
        faces = self.detect_faces(img)
        if not faces:
            return None
        
        # Sort by area of the bounding box
        largest_face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
        return largest_face

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine similarity between two embeddings.
        """
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        
        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0
            
        return float(dot_product / (norm_emb1 * norm_emb2))
