import asyncio
import threading
import subprocess
import sys
import os
import queue
import json
import time
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import tomlkit
import webbrowser

app = FastAPI()

# 文件路径
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.toml"
STATIC_DIR = SCRIPT_DIR / "static"

# 进程管理类
class ProcessManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.sync_queue = queue.Queue() # 线程安全队列
        self.active_websockets: List[WebSocket] = []
        self.is_running = False
        self.log_buffer: List[str] = []
        self.max_buffer_size = 5000

    async def broadcast(self, data: dict):
        """
        稳健的 JSON 广播：自动剔除已关闭连接，防止 IO 阻塞
        """
        msg = json.dumps(data)
        to_remove = []
        # 使用 list() 创建副本进行迭代，防止多协程并发修改导致 RuntimeError
        for ws in list(self.active_websockets):
            try:
                # 检查连接状态 (1 为 OPEN)
                if ws.client_state.value == 1:
                    await ws.send_text(msg)
                else:
                    to_remove.append(ws)
            except:
                to_remove.append(ws)
        
        for ws in to_remove:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)

    async def update_state(self, is_running: bool):
        self.is_running = is_running
        await self.broadcast({"type": "state", "is_running": is_running})

    async def add_log_batch(self, text: str):
        self.log_buffer.append(text)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)
        
        await self.broadcast({"type": "log", "content": text})

    def run_commands_sync(self, cmds: List[Any]):
        self.is_running = True
        # 立即通过队列通知消费者状态变更（由消费者协程广播）
        self.sync_queue.put({"type": "state", "is_running": True})

        try:
            for cmd in cmds:
                if not self.is_running: break
                
                self.process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    bufsize=0
                )
                
                def _read_io():
                    try:
                        # 增加读取块大小，提高大数据量下的吞吐
                        while True:
                            if not self.process or not self.process.stdout: break
                            chunk = self.process.stdout.read(1024)
                            if not chunk: break
                            
                            # 2. 多重解码逻辑方案
                            try:
                                # 优先尝试 UTF-8
                                text = chunk.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    # 针对 Windows 常见的 GBK 回退
                                    text = chunk.decode('gbk')
                                except UnicodeDecodeError:
                                    # 暴力兜底
                                    text = chunk.decode('utf-8', errors='replace')
                            
                            self.sync_queue.put({"type": "log", "content": text})
                    except Exception as e:
                        self.sync_queue.put({"type": "log", "content": f"\n[READ ERROR] {str(e)}\n"})
                
                io_thread = threading.Thread(target=_read_io)
                io_thread.daemon = True
                io_thread.start()
                
                self.process.wait()
                if self.process.returncode != 0 and self.is_running:
                    self.sync_queue.put({"type": "log", "content": f"\n[ERROR] Command failed with code {self.process.returncode}\n"})
                    break
        except Exception as e:
            self.sync_queue.put({"type": "log", "content": f"\n[EXCEPTION] {str(e)}\n"})
        finally:
            self.is_running = False
            self.process = None
            self.sync_queue.put({"type": "state", "is_running": False})
            self.sync_queue.put({"type": "log", "content": "\n[DONE] 所有任务已结束\n"})

    def start(self, cmds: List[Any]):
        if self.is_running:
            raise HTTPException(status_code=400, detail="任务正在运行中")
        thread = threading.Thread(target=self.run_commands_sync, args=(cmds,))
        thread.daemon = True
        thread.start()

    def stop(self):
        self.is_running = False
        # 清空待执行队列
        while not self.sync_queue.empty():
            try:
                self.sync_queue.get_nowait()
            except:
                break
                
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        self.process = None

manager = ProcessManager()

async def _log_consumer():
    """
    高效消费者：支持日志批处理和状态主动推送
    """
    log_accumulator = []
    last_send_time = time.time()
    
    while True:
        try:
            # 尽可能排空当前队列
            while not manager.sync_queue.empty():
                item = manager.sync_queue.get_nowait()
                
                if item["type"] == "state":
                    # 状态变更立即推送，并清空之前的日志积攒
                    if log_accumulator:
                        await manager.add_log_batch("".join(log_accumulator))
                        log_accumulator = []
                    await manager.update_state(item["is_running"])
                
                elif item["type"] == "log":
                    log_accumulator.append(item["content"])
                    
                    # 如果积攒的内容过多，立即发送
                    if len(log_accumulator) > 50:
                        await manager.add_log_batch("".join(log_accumulator))
                        log_accumulator = []
                        last_send_time = time.time()

            # 时间阈值检查：超过 50ms 必须发送一次积攒的日志
            if log_accumulator and (time.time() - last_send_time > 0.05):
                await manager.add_log_batch("".join(log_accumulator))
                log_accumulator = []
                last_send_time = time.time()
                
        except Exception as e:
            print(f"Log consumer error: {e}")
            
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_log_consumer())

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    manager.active_websockets.append(websocket)
    
    try:
        # 1. 立即发送“初始化报文”，同步任务状态和历史日志
        await websocket.send_text(json.dumps({
            "type": "init",
            "is_running": manager.is_running,
            "history": "".join(manager.log_buffer)
        }))
        
        # 2. 持续监听以保持连接活力
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in manager.active_websockets:
            manager.active_websockets.remove(websocket)

@app.get("/api/logs/status")
async def get_log_status():
    return {
        "is_running": manager.is_running,
        "has_logs": len(manager.log_buffer) > 0
    }

@app.post("/api/stop")
async def stop_task():
    manager.stop()
    return {"status": "success"}

class RawConfig(BaseModel):
    content: str

@app.get("/api/raw_config")
async def get_raw_config():
    try:
        config_path = Path("src/config.toml")
        if config_path.exists():
            return {"content": config_path.read_text(encoding="utf-8")}
        return {"content": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/raw_config")
async def save_raw_config(config: RawConfig):
    try:
        config_path = Path("src/config.toml")
        config_path.write_text(config.content, encoding="utf-8")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PickerRequest(BaseModel):
    mode: str  # "file" or "folder"
    title: str = "选择"
    filetypes: List[str] = []  # e.g., [".safetensors", ".pt"]

@app.post("/api/pick_path")
async def pick_path(req: PickerRequest):
    """
    弹出系统文件/文件夹选择对话框
    """
    import tkinter as tk
    from tkinter import filedialog
    
    def run_dialog():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if req.mode == "folder":
            path = filedialog.askdirectory(title=req.title)
        else:
            # 构建文件类型过滤器
            filetypes = []
            if req.filetypes:
                ext_str = " ".join(f"*{ext}" for ext in req.filetypes)
                filetypes.append((f"支持的文件 ({ext_str})", ext_str))
            filetypes.append(("所有文件", "*.*"))
            path = filedialog.askopenfilename(title=req.title, filetypes=filetypes)
        
        root.destroy()
        return path
    
    # 在主线程中运行 tkinter（避免线程问题）
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_dialog)
        path = future.result()
    
    if path:
        # 智能路径转换：如果是当前目录下的文件，转换为相对路径
        try:
            p = Path(path).resolve()
            cwd = Path.cwd().resolve()
            # 尝试获取相对路径
            rel = p.relative_to(cwd)
            # 强制使用正斜杠，并添加 ./ 前缀以明确是相对路径
            path = f"./{rel.as_posix()}"
        except ValueError:
            # 不在当前目录下（或跨盘符），使用绝对路径，统一转为正斜杠
            path = Path(path).as_posix()

    return {"path": path or ""}

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools_handler():
    return JSONResponse(content={}, status_code=404)

# 硬编码模板
TEMPLATES = {
    "Qwen-Image": {
        "general": {
            "resolution": [1024, 1024],
            "caption_extension": ".txt",
            "batch_size": 1,
            "num_repeats": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [{
            "image_directory": "./dataset/image_dir",
            "cache_directory": "./dataset/cache_directory",
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
            "resolution": [1024, 1024],
            "caption_extension": ".txt",
            "batch_size": 1,
            "num_repeats": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [{
            "image_directory": "./dataset/image_dir",
            "cache_directory": "./dataset/cache_directory",
            "control_directory": "./dataset/control_dir",
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
    }
}

# 辅助函数：读取 TOML
def read_toml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())

import tempfile
import shutil

# 辅助函数：写入 TOML (原子写入，防止损坏)
def write_toml(path: Path, data: Dict):
    # 先尝试序列化，如果失败不会损坏文件
    try:
        content = tomlkit.dumps(data)
    except Exception as e:
        print(f"Error serializing TOML: {e}")
        raise e

    # 写入临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=path.parent, encoding="utf-8", suffix=".tmp") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        # 原子替换
        if path.exists():
            # Windows 下 os.replace 可能在目标存在时失败/行为不一致，但在 Python 3.3+ 应该是原子的
            # 为了保险起见，可以先备份？或者直接 replace
            os.replace(tmp_path, path)
        else:
            os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise e

class SaveData(BaseModel):
    is_new: bool
    old_name: str = ""  # 如果是修改已有任务，旧的名字
    fixed_params: Dict[str, Any]
    template_params: Dict[str, Any]
    train_type: str

# 验证数字格式
def validate_number(key: str, value: Any, is_float: bool = False):
    if value == "" or value is None:
        return value
    if isinstance(value, str):
        # 允许纯数字，或者带小数点的数字
        # 严禁出现 + - e 等符号
        cleaned = value.strip()
        if not cleaned: return value
        
        # 检查非法字符
        if is_float:
            # 允许科学计数法
            allowed = "0123456789.eE+-"
        else:
            allowed = "0123456789"

        if any(c not in allowed for c in cleaned):
             raise HTTPException(status_code=400, detail=f"字段 {key} 包含非法字符，只允许数字{ '和小数点及科学计数法' if is_float else ''}")
        
        # 尝试转换
        try:
            if is_float:
                return float(cleaned)
            else:
                if '.' in cleaned:
                     raise ValueError("Int field has dot")
                return int(cleaned)
        except ValueError:
             raise HTTPException(status_code=400, detail=f"字段 {key} 格式错误")
    
    # 如果已经是数字，检查类型
    if isinstance(value, (int, float)):
        return value

    return value

@app.get("/api/tasks")
async def get_tasks():
    config = read_toml(CONFIG_PATH)
    return config.get("task", [])

@app.get("/api/templates")
async def get_templates():
    return TEMPLATES

@app.get("/api/config/{name}")
async def get_task_config(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)
    
    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 读取任务对应的 toml 文件
    task_toml_path = SCRIPT_DIR / f"{name}.toml"
    template_params = {}
    try:
        template_params = read_toml(task_toml_path)
    except Exception as e:
        print(f"Error reading task toml {task_toml_path}: {e}")
    
    # 容错处理：如果 toml 文件不存在或为空，或读取失败，使用默认模板
    if not template_params:
        model_ver = str(task_item.get("model_version", "")).lower()
        if "edit" in model_ver:
            template_params = TEMPLATES["Qwen-Image-Edit-2511"]
        else:
            template_params = TEMPLATES["Qwen-Image"]
    
    return {
        "fixed_params": task_item,
        "template_params": template_params
    }

@app.post("/api/save")
async def save_task(data: SaveData):
    try:
        output_name = data.fixed_params.get("output_name")
        if not output_name:
            raise HTTPException(status_code=400, detail="output_name 不能为空")
        
        # 后端严格校验
        # 1. fixed_params 校验
        fixed_int_keys = ['max_train_epochs', 'save_every_n_epochs', 'sample_every_n_epochs', 'network_dim', 'blocks_to_swap', 'network_alpha', 'loraplus_lr_ratio', 'frame_count']
        fixed_float_keys = ['learning_rate']
        
        for k in fixed_int_keys:
            if k in data.fixed_params:
                data.fixed_params[k] = validate_number(k, data.fixed_params[k], is_float=False)
        
        for k in fixed_float_keys:
            if k in data.fixed_params:
                data.fixed_params[k] = validate_number(k, data.fixed_params[k], is_float=True)

        config = read_toml(CONFIG_PATH)
        tasks = config.get("task", [])
        
        # 检查重名
        if data.is_new:
            if any(t.get("output_name") == output_name for t in tasks):
                raise HTTPException(status_code=400, detail="任务名已存在")
            tasks.append(data.fixed_params)
        else:
            # 修改模式
            found = False
            for idx, t in enumerate(tasks):
                if t.get("output_name") == data.old_name:
                    tasks[idx] = data.fixed_params
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail="未找到要修改的任务")
            
            # 如果改名了，删除旧的 toml 文件
            if data.old_name != output_name:
                old_toml = SCRIPT_DIR / f"{data.old_name}.toml"
                if old_toml.exists():
                    old_toml.unlink()

        # 所有的逻辑处理完毕，数据准备好后，再执行写入
        # 任何之前的错误都会跳到 except，不会执行下面的写入
        
        # 优化 TOML 写入逻辑：
        # 1. 确保任务块始终位于文件末尾（通过删除后重新赋值）
        # 2. 如果没有任务，彻底移除 [[task]] 键，避免残留 task = []
        
        if "task" in config:
            del config["task"]
            
        if tasks:
            # 强制转换为 TOML AOT 格式，确保输出为 [[task]]
            aot = tomlkit.aot()
            for t in tasks:
                # 必须转换为普通字典，否则 InlineTable 会导致 tomlkit 报错或保持行内格式
                aot.append(dict(t))
            config["task"] = aot
        
        write_toml(CONFIG_PATH, config)
        
        # 保存任务专属的 toml
        task_toml_path = SCRIPT_DIR / f"{output_name}.toml"
        write_toml(task_toml_path, data.template_params)
        
        return {"status": "success"}

    except Exception as e:
        # 捕获所有异常，确保前端收到错误，且文件未被修改（因为 write_toml 是原子的，且在最后执行）
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.delete("/api/tasks/{name}")
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
    
    # 删除关联文件
    task_toml = SCRIPT_DIR / f"{name}.toml"
    if task_toml.exists():
        task_toml.unlink()
        
    return {"status": "success"}

@app.post("/api/cache/{name}")
async def run_cache(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)
    
    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")
    
    cache_config = config.get("cache", {})
    model_version = task_item.get("model_version", "edit-2511")
    dataset_config = task_item.get("dataset_config", "")
    text_encoder = task_item.get("text_encoder", "")
    vae_path = task_item.get("vae", "")
    
    # 获取当前 Python 解释器路径
    python_exe = sys.executable

    cmds = []
    
    # 1. Latents Cache Command
    latent_cmd = [
        python_exe, "-m", "accelerate.commands.launch", 
        "./qwen_image_cache_latents.py",
        f"--dataset_config={dataset_config}",
        f"--model_version={model_version}"
    ]
    if vae_path: latent_cmd.append(f"--vae={vae_path}")
    if cache_config.get("vae_tiling"): latent_cmd.append("--vae_tiling")
    if cache_config.get("vae_chunk_size"): latent_cmd.append(f"--vae_chunk_size={cache_config['vae_chunk_size']}")
    if cache_config.get("vae_spatial_tile_sample_min_size"): latent_cmd.append(f"--vae_spatial_tile_sample_min_size={cache_config['vae_spatial_tile_sample_min_size']}")
    
    cmds.append(latent_cmd)

    # 2. Text Encoder Cache Command
    te_cmd = [
        python_exe, "-m", "accelerate.commands.launch", 
        "./qwen_image_cache_text_encoder_outputs.py",
        f"--dataset_config={dataset_config}",
        f"--text_encoder={text_encoder}",
        f"--model_version={model_version}",
        f"--batch_size={cache_config.get('batch_size', 16)}"
    ]
    if cache_config.get("fp8_vl"): te_cmd.append("--fp8_vl")
    
    cmds.append(te_cmd)

    print(f"[DEBUG] Cache Commands: {cmds}")
    try:
        manager.start(cmds)
        return {"status": "success", "message": "Caching started"}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train/{name}")
async def run_train(name: str):
    config = read_toml(CONFIG_PATH)
    tasks = config.get("task", [])
    task_item = next((t for t in tasks if t.get("output_name") == name), None)
    
    if not task_item:
        raise HTTPException(status_code=404, detail="Task not found")
        
    global_config = config.get("global_config", {})
    python_exe = sys.executable

    # 基础命令架构
    cmd = [
        python_exe, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process=8",
        "--mixed_precision=bf16",
        "--downcast_bf16",
        "./qwen_image_train_network.py"
    ]

    # 参数映射列表
    # (键名, 命令行参数名, 是否是 Flag)
    param_map = [
        # 1. 核心模型/任务参数 (来自 task_item)
        ("dit", "--dit", False),
        ("vae", "--vae", False),
        ("text_encoder", "--text_encoder", False),
        ("dataset_config", "--dataset_config", False),
        ("seed", "--seed", False),
        ("learning_rate", "--learning_rate", False),
        ("timestep_sampling", "--timestep_sampling", False),
        ("network_dim", "--network_dim", False),
        ("network_alpha", "--network_alpha", False),
        ("blocks_to_swap", "--blocks_to_swap", False),
        ("output_name", "--output_name", False),
        ("output_dir", "--output_dir", False),
        ("model_version", "--model_version", False),
        ("max_train_epochs", "--max_train_epochs", False),
        ("save_every_n_epochs", "--save_every_n_epochs", False),
        ("sample_every_n_epochs", "--sample_every_n_epochs", False),
        ("sample_prompts", "--sample_prompts", False),
        
        # 2. 全局/训练优化参数 (来自 global_config)
        ("logging_dir", "--logging_dir", False),
        ("flash_attn", "--flash_attn", True),
        ("split_attn", "--split_attn", True),
        ("guidance_scale", "--guidance_scale", False),
        ("gradient_checkpointing", "--gradient_checkpointing", True),
        ("network_module", "--network_module", False),
        ("lr_scheduler", "--lr_scheduler", False),
        ("lr_scheduler_num_cycles", "--lr_scheduler_num_cycles", False),
        ("lr_decay_steps", "--lr_decay_steps", False),
        ("lr_scheduler_min_lr_ratio", "--lr_scheduler_min_lr_ratio", False),
        ("cuda_allow_tf32", "--cuda_allow_tf32", True),
        ("cuda_cudnn_benchmark", "--cuda_cudnn_benchmark", True),
        ("mixed_precision", "--mixed_precision", False),
        ("fp8_base", "--fp8_base", True),
        ("fp8_scaled", "--fp8_scaled", True),
        ("persistent_data_loader_workers", "--persistent_data_loader_workers", True),
        ("use_pinned_memory_for_block_swap", "--use_pinned_memory_for_block_swap", True),
        ("compile", "--compile", True),
        ("compile_backend", "--compile_backend", False),
        ("compile_mode", "--compile_mode", False),
        ("compile_dynamic", "--compile_dynamic", False),
        ("compile_cache_size_limit", "--compile_cache_size_limit", False),
        ("img_in_txt_in_offloading", "--img_in_txt_in_offloading", True),
        ("optimizer_type", "--optimizer_type", False),
    ]

    # 动态组装参数
    assembled_args = [
        python_exe, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process=8",
        "--mixed_precision=bf16",
        "--downcast_bf16",
        "./qwen_image_train_network.py"
    ]
    
    for key, arg, is_flag in param_map:
        # 优先从 task_item 取，取不到从 global_config 取
        val = task_item.get(key)
        if val is None or val == "":
            val = global_config.get(key)
            
        if val is None or val == "":
            continue
            
        if is_flag:
            if val is True:
                assembled_args.append(arg)
        else:
            # 严格遵循 --参数名=参数值 格式
            assembled_args.append(f"{arg}={val}")

    # 特殊处理：loraplus_lr_ratio -> --network_args loraplus_lr_ratio=X
    lora_ratio = task_item.get("loraplus_lr_ratio")
    if lora_ratio is not None and lora_ratio != "":
        assembled_args.append("--network_args")
        assembled_args.append(f"loraplus_lr_ratio={lora_ratio}")

    print(f"[DEBUG] Train Command: {assembled_args}")

    try:
        manager.start([assembled_args]) 
        return {"status": "success", "message": "Training started"}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

# 获取任务目录下的第一张图片作为缩略图
@app.get("/api/thumbnail")
async def get_thumbnail(path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    
    # 稳健的路径解析逻辑
    p = Path(path)
    
    # 优先级 1: 绝对路径
    if p.is_absolute() and p.exists():
        abs_p = p
    else:
        # 优先级 2: 相对于项目根目录 (CWD)
        abs_p = (Path(os.getcwd()) / p).resolve()
        if not abs_p.exists():
            # 优先级 3: 相对于脚本目录
            abs_p = (SCRIPT_DIR / p).resolve()

    if not abs_p.exists() or not abs_p.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    # 支持的图片扩展名
    exts = ('.jpg', '.png', '.jpeg', '.webp', '.bmp')
    try:
        # 仅获取该层级的图片，不递归（防止太慢）
        for f in abs_p.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                return FileResponse(f)
    except Exception as e:
        print(f"Thumbnail error: {e}")

    raise HTTPException(status_code=404, detail="No images found")

@app.on_event("startup")
async def startup_event():
    target_url = "http://127.0.0.1:9980"
    await asyncio.sleep(0.5)
    webbrowser.open(target_url)

# 静态文件服务
@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9980)