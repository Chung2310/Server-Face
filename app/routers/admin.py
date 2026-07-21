import math
import psutil
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile, File, Form

from app.config import settings
from app.database import get_db, require_database
from app.schemas.face import (
    AdminLoginRequest, AdminMeResponse, RegisterFaceResponse,
    RegistrationStatusResponse, DeleteRegistrationResponse, FaceRegistrationInfo,
    AttendanceLogEntry, AttendanceLogsResponse,
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


# ── Attendance logs endpoints ──

@router.get(
    "/attendance-logs",
    response_model=AttendanceLogsResponse,
    summary="Lịch sử chấm công - lịch sử xác thực khuôn mặt",
)
async def get_attendance_logs(
    username: str = Depends(get_current_admin),
    user_id: Optional[str] = Query(None, description="Lọc theo mã nhân viên (chứa)"),
    verified: Optional[bool] = Query(None, description="Lọc theo kết quả: true=thành công, false=thất bại"),
    date_from: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    date_to:   Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)"),
    year:      Optional[int] = Query(None, description="Lọc theo năm"),
    month:     Optional[int] = Query(None, ge=1, le=12, description="Lọc theo tháng (1-12)"),
    day:       Optional[int] = Query(None, ge=1, le=31, description="Lọc theo ngày trong tháng (1-31)"),
    hour_from: Optional[int] = Query(None, ge=0, le=23, description="Giờ bắt đầu (0-23)"),
    hour_to:   Optional[int] = Query(None, ge=0, le=23, description="Giờ kết thúc (0-23)"),
    page:      int           = Query(1, ge=1, description="Trang hiện tại"),
    limit:     int           = Query(20, ge=1, le=100, description="Số bản ghi mỗi trang"),
):
    """
    Trả về lịch sử xác thực khuôn mặt (chấm công) có filter và phân trang.
    Hỗ trợ lọc theo: mã nhân viên, kết quả, khoảng ngày, năm, tháng, ngày, giờ.
    """
    db = require_database()

    # Build MongoDB filter query
    query: dict = {}

    if user_id:
        query["user_id"] = {"$regex": user_id, "$options": "i"}

    if verified is not None:
        query["verified"] = verified

    # Time range filter using $gte / $lte on timestamp
    time_filter: dict = {}
    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            time_filter["$gte"] = dt_from
        except ValueError:
            raise HTTPException(status_code=400, detail="date_from phải có định dạng YYYY-MM-DD")
    if date_to:
        try:
            from datetime import timedelta
            dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            time_filter["$lt"] = dt_to
        except ValueError:
            raise HTTPException(status_code=400, detail="date_to phải có định dạng YYYY-MM-DD")
    if time_filter:
        query["timestamp"] = time_filter

    # Year/Month/Day/Hour filters using aggregation pipeline when needed
    use_pipeline = any(v is not None for v in [year, month, day, hour_from, hour_to])

    if use_pipeline:
        pipeline_match: dict = {**query}
        pipeline = [
            {"$addFields": {
                "_year":  {"$year": {"date": "$timestamp", "timezone": "+07:00"}},
                "_month": {"$month": {"date": "$timestamp", "timezone": "+07:00"}},
                "_day":   {"$dayOfMonth": {"date": "$timestamp", "timezone": "+07:00"}},
                "_hour":  {"$hour": {"date": "$timestamp", "timezone": "+07:00"}},
            }},
        ]
        if pipeline_match:
            pipeline.insert(0, {"$match": pipeline_match})

        extra_match: dict = {}
        if year:       extra_match["_year"]  = year
        if month:      extra_match["_month"] = month
        if day:        extra_match["_day"]   = day
        if hour_from is not None and hour_to is not None:
            extra_match["_hour"] = {"$gte": hour_from, "$lte": hour_to}
        elif hour_from is not None:
            extra_match["_hour"] = {"$gte": hour_from}
        elif hour_to is not None:
            extra_match["_hour"] = {"$lte": hour_to}

        if extra_match:
            pipeline.append({"$match": extra_match})

        pipeline += [
            {"$sort": {"timestamp": -1}},
            {"$facet": {
                "total": [{"$count": "count"}],
                "data":  [{"$skip": (page - 1) * limit}, {"$limit": limit}],
            }},
        ]

        cursor = db.verification_logs.aggregate(pipeline)
        result_docs = await cursor.to_list(length=1)
        if not result_docs:
            return AttendanceLogsResponse(total=0, page=page, limit=limit, total_pages=0, data=[])
        facet = result_docs[0]
        total = facet["total"][0]["count"] if facet["total"] else 0
        docs = facet["data"]
    else:
        total = await db.verification_logs.count_documents(query)
        cursor = db.verification_logs.find(query).sort("timestamp", -1).skip((page - 1) * limit).limit(limit)
        docs = await cursor.to_list(length=limit)

    entries = []
    for doc in docs:
        entries.append(AttendanceLogEntry(
            id=str(doc["_id"]),
            user_id=doc.get("user_id", ""),
            verified=doc.get("verified", False),
            similarity=doc.get("similarity"),
            reason=doc.get("reason", ""),
            timestamp=doc.get("timestamp"),
            ip_address=doc.get("ip_address"),
            device_info=doc.get("device_info"),
        ))

    total_pages = math.ceil(total / limit) if total > 0 else 0
    return AttendanceLogsResponse(
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        data=entries,
    )


@router.get(
    "/attendance-logs/stats",
    summary="Thống kê chấm công tổng hợp",
)
async def get_attendance_stats(username: str = Depends(get_current_admin)):
    """Trả về thống kê nhanh: tổng lượt, thành công, thất bại hôm nay và toàn bộ."""
    db = require_database()
    from datetime import timedelta
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_end   = today_start + timedelta(days=1)

    total_all     = await db.verification_logs.count_documents({})
    success_all   = await db.verification_logs.count_documents({"verified": True})
    today_all     = await db.verification_logs.count_documents({"timestamp": {"$gte": today_start, "$lt": today_end}})
    today_success = await db.verification_logs.count_documents({"verified": True, "timestamp": {"$gte": today_start, "$lt": today_end}})

    return {
        "total_all":       total_all,
        "success_all":     success_all,
        "failed_all":      total_all - success_all,
        "today_all":       today_all,
        "today_success":   today_success,
        "today_failed":    today_all - today_success,
        "success_rate":    round(success_all / total_all * 100, 1) if total_all > 0 else 0.0,
    }
