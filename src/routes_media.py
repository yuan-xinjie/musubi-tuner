"""
媒体资源路由：缩略图（图片 + 视频）预览
"""
import os
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config_utils import SCRIPT_DIR

router = APIRouter()

IMG_EXTS = ('.jpg', '.png', '.jpeg', '.webp', '.bmp')
VID_EXTS = ('.mp4', '.webm', '.gif', '.avi', '.mov')

# ─── 路径解析辅助 ────────────────────────────────────────────

def _resolve_dir(path: str) -> Path:
    """按优先级解析目录路径：绝对 → 相对 CWD → 相对脚本"""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p

    abs_p = (Path(os.getcwd()) / p).resolve()
    if abs_p.exists():
        return abs_p

    abs_p = (SCRIPT_DIR / p).resolve()
    if abs_p.exists():
        return abs_p

    return abs_p  # 即使不存在也返回，由调用方判断

# ─── 扫描首个媒体文件 ────────────────────────────────────────

def _scan_first_media(abs_p: Path):
    """在目录中扫描首个图片和首个视频文件"""
    first_img = None
    first_vid = None
    try:
        for f in abs_p.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                if ext in IMG_EXTS and not first_img:
                    first_img = f
                if ext in VID_EXTS and not first_vid:
                    first_vid = f
                if first_img:
                    break
    except Exception as e:
        print(f"Thumbnail scan error: {e}")
    return first_img, first_vid

# ─── 缩略图文件返回 ──────────────────────────────────────────

@router.get("/api/thumbnail")
async def get_thumbnail(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")

    abs_p = _resolve_dir(path)
    if not abs_p.exists() or not abs_p.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    first_img, first_vid = None, None
    try:
        for f in abs_p.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                if ext in IMG_EXTS and not first_img:
                    first_img = f
                if ext in VID_EXTS and not first_vid:
                    first_vid = f
                if first_img:
                    return FileResponse(first_img)
        if first_vid:
            return FileResponse(first_vid)
    except Exception as e:
        print(f"Thumbnail error: {e}")

    raise HTTPException(status_code=404, detail="No media found")

# ─── 缩略图元信息（图片 or 视频） ───────────────────────────

@router.get("/api/thumbnail_meta")
async def get_thumbnail_meta(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")

    abs_p = _resolve_dir(path)
    if not abs_p.exists() or not abs_p.is_dir():
        raise HTTPException(status_code=404)

    first_img, first_vid = _scan_first_media(abs_p)
    target = first_img or first_vid
    if target:
        return {
            "url": f"/api/thumbnail?path={urllib.parse.quote(path)}",
            "is_video": target.suffix.lower() in VID_EXTS
        }

    raise HTTPException(status_code=404)
