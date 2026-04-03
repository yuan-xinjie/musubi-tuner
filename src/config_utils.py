"""
配置工具模块：常量定义、TOML 读写、模板定义、数据模型、校验函数
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import tomlkit
from pydantic import BaseModel
from fastapi import HTTPException

# ─── 路径常量 ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.toml"
STATIC_DIR = SCRIPT_DIR / "static"

# ─── 数据集模板 ──────────────────────────────────────────────
TEMPLATES = {
    "Qwen-Image": {
        "general": {
            "caption_extension": ".txt",
            "batch_size": 1,
            "num_repeats": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [{
            "resolution": [1024, 1024],
            "image_directory": "./dataset/image",
            "cache_directory": "./dataset/image/cache",
        }],
        "samples": [{
            "prompt": "a photo of a dog",
            "width": 1024,
            "height": 1024,
            "sample_steps": 20,
            "guidance_scale": 3.0,
            "seed": 42,
            "frame_count": 1,
            "discrete_flow_shift": 3.0,
        }]
    },
    "Qwen-Image-Edit-2511": {
        "general": {
            "caption_extension": ".txt",
            "batch_size": 1,
            "num_repeats": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [{
            "resolution": [1024, 1024],
            "image_directory": "./dataset/target",
            "cache_directory": "./dataset/target/cache",
            "control_directory": "./dataset/control",
            "qwen_image_edit_control_resolution": [1024, 1024],
        }],
        "samples": [{
            "prompt": "a photo of a dog",
            "width": 1024,
            "height": 1024,
            "sample_steps": 20,
            "guidance_scale": 3.0,
            "seed": 42,
            "discrete_flow_shift": 3.0,
            "control_image_path": ["./dataset/ctrl/1.png"],
        }]
    },
    "Wan2.2": {
        "general": {
            "caption_extension": ".txt",
            "batch_size": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "num_repeats": 10,
        },
        "datasets": [{
            "resolution": [512, 512],
            "video_directory": "./dataset/video",
            "cache_directory": "./dataset/video/cache",
            "target_frames": [1, 13, 25],
            "frame_sample": 4,
            "frame_extraction": "uniform",
            "max_frames": 81,
        }]
    }
}

# ─── Pydantic 数据模型 ───────────────────────────────────────

class RawConfig(BaseModel):
    content: str

class PickerRequest(BaseModel):
    mode: str  # "file" or "folder"
    title: str = "选择"
    filetypes: List[str] = []

class SaveData(BaseModel):
    is_new: bool
    old_name: str = ""
    fixed_params: Dict[str, Any]
    template_params: Dict[str, Any]
    train_type: str

# ─── TOML 读写 ───────────────────────────────────────────────

def read_toml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())

def write_toml(path: Path, data: Dict):
    """原子写入 TOML，防止中途失败导致文件损坏"""
    try:
        content = tomlkit.dumps(data)
    except Exception as e:
        print(f"Error serializing TOML: {e}")
        raise e

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, dir=path.parent,
            encoding="utf-8", suffix=".tmp"
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise e

# ─── 校验函数 ────────────────────────────────────────────────

def validate_number(key: str, value: Any, is_float: bool = False):
    """校验并转换数字类型字段"""
    if value == "" or value is None:
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return value

        allowed = "0123456789.eE+-" if is_float else "0123456789"
        if any(c not in allowed for c in cleaned):
            raise HTTPException(
                status_code=400,
                detail=f"字段 {key} 包含非法字符，只允许数字{'和小数点及科学计数法' if is_float else ''}"
            )

        try:
            if is_float:
                return float(cleaned)
            else:
                if '.' in cleaned:
                    raise ValueError("Int field has dot")
                return int(cleaned)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"字段 {key} 格式错误")

    if isinstance(value, (int, float)):
        return value
    return value

# ─── 任务类型判断 ─────────────────────────────────────────────

def is_wan_task(task_item: Dict) -> bool:
    """判断任务是否属于 Wan2.2 类型"""
    return (
        task_item.get("train_type") == "Wan2.2"
        or "task" in task_item
    )
