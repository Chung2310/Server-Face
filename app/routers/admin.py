import secrets
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.config import settings
from app.database import db

router = APIRouter(prefix="/admin", tags=["Admin"])
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Xác thực tài khoản quản trị viên qua HTTP Basic Auth."""
    correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.ADMIN_USERNAME.encode("utf-8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.ADMIN_PASSWORD.encode("utf-8")
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không đúng.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@router.get("/metrics", summary="Lấy thông số giám sát hệ thống (yêu cầu xác thực quản trị)")
async def get_metrics(username: str = Depends(verify_admin)):
    """
    Trả về thông số giám sát thời gian thực của hệ thống:
    - Mức sử dụng CPU và RAM.
    - Tổng số nhân viên đã đăng ký trong cơ sở dữ liệu khuôn mặt.
    - Thời điểm truy vấn.
    """
    # System metrics (non-blocking: interval=None trả về giá trị cached ngay lập tức)
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()

    # Database metrics
    try:
        total_employees = await db.face_registry.count_documents({})
        db_status = "connected"
    except Exception:
        total_employees = 0
        db_status = "disconnected"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": {
            "cpu_percent": cpu_percent,
            "memory_total_mb": round(memory.total / 1024 / 1024, 1),
            "memory_used_mb": round(memory.used / 1024 / 1024, 1),
            "memory_percent": memory.percent,
        },
        "database": {
            "status": db_status,
            "total_registered_employees": total_employees,
        },
        "service": {
            "name": settings.APP_NAME,
            "model": settings.MODEL_NAME,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        }
    }
