"""
任务管理路由：CRUD 操作、配置读写、文件选择器
"""
import os
import concurrent.futures
from pathlib import Path
from typing import List

import tomlkit
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config_utils import (
    SCRIPT_DIR, CONFIG_PATH, TEMPLATES,
    RawConfig, PickerRequest, SaveData,
    read_toml, write_toml, validate_number, is_wan_task
)

router = APIRouter()

# ─── 文件/文件夹选择器 ───────────────────────────────────────

@router.post("/api/pick_path")
async def pick_path(req: PickerRequest):
    """弹出系统文件/文件夹选择对话框"""
    import tkinter as tk
    from tkinter import filedialog

    def run_dialog():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        if req.mode == "folder":
            path = filedialog.askdirectory(title=req.title)
        else:
            filetypes = []
            if req.filetypes:
                ext_str = " ".join(f"*{ext}" for ext in req.filetypes)
                filetypes.append((f"支持的文件 ({ext_str})", ext_str))
            filetypes.append(("所有文件", "*.*"))
            path = filedialog.askopenfilename(title=req.title, filetypes=filetypes)

        root.destroy()
        return path

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_dialog)
        path = future.result()

    if path:
        try:
            p = Path(path).resolve()
            cwd = Path.cwd().resolve()
            rel = p.relative_to(cwd)
            path = f"./{rel.as_posix()}"
        except ValueError:
            path = Path(path).as_posix()

    return {"path": path or ""}

# ─── Chrome DevTools 静默处理 ────────────────────────────────

@router.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools_handler():
    return JSONResponse(content={}, status_code=404)

# ─── 原始配置编辑 ────────────────────────────────────────────

@router.get("/api/raw_config")
async def get_raw_config():
    try:
        if CONFIG_PATH.exists():
            return {"content": CONFIG_PATH.read_text(encoding="utf-8")}
        return {"content": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/raw_config")
async def save_raw_config(config: RawConfig):
    try:
        CONFIG_PATH.write_text(config.content, encoding="utf-8")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── 模板 & 任务列表 ────────────────────────────────────────

@router.get("/api/templates")
async def get_templates():
    return TEMPLATES

@router.get("/api/tasks")
async def get_tasks():
    config = read_toml(CONFIG_PATH)
    return config.get("task", [])

# ─── 单任务配置读取 ──────────────────────────────────────────

@router.get("/api/config/{name}")
async def get_task_config(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)

    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")

    task_toml_path = SCRIPT_DIR / f"{name}.toml"
    template_params = {}
    try:
        template_params = read_toml(task_toml_path)
    except Exception as e:
        print(f"Error reading task toml {task_toml_path}: {e}")

    if not template_params:
        model_ver = str(task_item.get("model_version", "")).lower()
        if "edit" in model_ver:
            template_params = TEMPLATES["Qwen-Image-Edit-2511"]
        elif is_wan_task(task_item):
            template_params = TEMPLATES["Wan2.2"]
        else:
            template_params = TEMPLATES["Qwen-Image"]

    return {
        "fixed_params": task_item,
        "template_params": template_params
    }

# ─── 保存任务 ────────────────────────────────────────────────

@router.post("/api/save")
async def save_task(data: SaveData):
    try:
        output_name = data.fixed_params.get("output_name")
        if not output_name:
            raise HTTPException(status_code=400, detail="output_name 不能为空")

        # 后端严格校验
        fixed_int_keys = [
            'max_train_epochs', 'save_every_n_epochs', 'sample_every_n_epochs',
            'network_dim', 'blocks_to_swap', 'network_alpha', 'loraplus_lr_ratio',
            'frame_count'
        ]
        fixed_float_keys = ['learning_rate']

        for k in fixed_int_keys:
            if k in data.fixed_params:
                data.fixed_params[k] = validate_number(k, data.fixed_params[k], is_float=False)
        for k in fixed_float_keys:
            if k in data.fixed_params:
                data.fixed_params[k] = validate_number(k, data.fixed_params[k], is_float=True)

        config = read_toml(CONFIG_PATH)
        tasks = config.get("task", [])

        # 记录 train_type
        if hasattr(data, "train_type") and data.train_type:
            data.fixed_params["train_type"] = data.train_type

        # 检查重名 / 更新
        if data.is_new:
            if any(t.get("output_name") == output_name for t in tasks):
                raise HTTPException(status_code=400, detail="任务名已存在")
            tasks.append(data.fixed_params)
        else:
            found = False
            for idx, t in enumerate(tasks):
                if t.get("output_name") == data.old_name:
                    tasks[idx] = data.fixed_params
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail="未找到要修改的任务")

            if data.old_name != output_name:
                old_toml = SCRIPT_DIR / f"{data.old_name}.toml"
                if old_toml.exists():
                    old_toml.unlink()

        # 写入主配置
        if "task" in config:
            del config["task"]
        if tasks:
            aot = tomlkit.aot()
            for t in tasks:
                aot.append(dict(t))
            config["task"] = aot
        write_toml(CONFIG_PATH, config)

        # 写入任务专属 toml
        task_toml_path = SCRIPT_DIR / f"{output_name}.toml"
        write_toml(task_toml_path, data.template_params)

        return {"status": "success"}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ─── 删除任务 ────────────────────────────────────────────────

@router.delete("/api/tasks/{name}")
async def delete_task(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])

    new_tasks = [t for t in tasks if t.get("output_name") != name]
    if len(new_tasks) == len(tasks):
        raise HTTPException(status_code=404, detail="任务不存在")

    if "task" in config:
        del config["task"]
    if new_tasks:
        aot = tomlkit.aot()
        for t in new_tasks:
            aot.append(dict(t))
        config["task"] = aot
    write_toml(CONFIG_PATH, config)

    task_toml = SCRIPT_DIR / f"{name}.toml"
    if task_toml.exists():
        task_toml.unlink()

    return {"status": "success"}
