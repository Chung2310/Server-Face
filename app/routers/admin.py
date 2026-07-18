import psutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, File, Form
from typing import List

from app.config import settings
from app.database import get_db, require_database
from app.schemas.face import (
    AdminLoginRequest, AdminMeResponse, RegisterFaceResponse,
    RegistrationStatusResponse, DeleteRegistrationResponse, FaceRegistrationInfo,
)
from app.services.admin_auth import (
    verify_password, create_session, delete_session, get_current_admin,
)
from app.routers.face import (
    register_face_from_image, get_registration_status,
    delete_registration, list_registrations,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/login", response_model=AdminMeResponse, summary="Đăng nhập quản trị viên (session cookie)")
async def login(payload: AdminLoginRequest, response: Response):
    db = require_database()
    admin = await db.admins.find_one({"username": payload.username})
    if (
        admin is None
        or not admin.get("active", False)
        or not verify_password(payload.password, admin.get("password_hash", ""))
    ):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng.")

    token, _expires_at = await create_session(db, payload.username)
    response.set_cookie(
        key=settings.ADMIN_SESSION_COOKIE,
        value=token,
        max_age=settings.ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.ADMIN_COOKIE_SECURE,
        path="/",
    )
    return AdminMeResponse(username=payload.username)


@router.post("/logout", summary="Đăng xuất và hủy phiên quản trị")
async def logout(request: Request, response: Response):
    db = require_database()
    token = request.cookies.get(settings.ADMIN_SESSION_COOKIE)
    if token:
        await delete_session(db, token)
    response.delete_cookie(key=settings.ADMIN_SESSION_COOKIE, path="/")
    return {"detail": "Đã đăng xuất."}


@router.get("/me", response_model=AdminMeResponse, summary="Khôi phục phiên đăng nhập hiện tại")
async def me(username: str = Depends(get_current_admin)):
    return AdminMeResponse(username=username)


@router.get("/metrics", summary="Lấy thông số giám sát hệ thống (yêu cầu phiên quản trị)")
async def get_metrics(username: str = Depends(get_current_admin)):
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
    db = get_db()
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


# ── Session-protected face registry endpoints for the admin UI ──

@router.get("/faces", response_model=List[FaceRegistrationInfo], summary="Danh sách nhân viên đã đăng ký khuôn mặt")
async def admin_list_faces(username: str = Depends(get_current_admin)):
    return await list_registrations()


@router.post("/faces", response_model=RegisterFaceResponse, summary="Đăng ký khuôn mặt từ giao diện quản trị")
async def admin_register_face(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(get_current_admin),
):
    contents = await file.read()
    return await register_face_from_image(user_id, contents)


@router.get("/faces/{user_id}", response_model=RegistrationStatusResponse, summary="Trạng thái đăng ký của một nhân viên")
async def admin_face_status(user_id: str, username: str = Depends(get_current_admin)):
    return await get_registration_status(user_id)


@router.delete("/faces/{user_id}", response_model=DeleteRegistrationResponse, summary="Xóa đăng ký khuôn mặt")
async def admin_delete_face(user_id: str, username: str = Depends(get_current_admin)):
    return await delete_registration(user_id)
