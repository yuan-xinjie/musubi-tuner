"""
Musubi-Tuner WebUI 主入口
职责：FastAPI 应用初始化、WebSocket 日志、路由注册、静态文件服务
"""
import asyncio
import json
import time
import webbrowser

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from config_utils import STATIC_DIR
from process_manager import ProcessManager

# 路由模块
import routes_task
import routes_runner
import routes_media

# ─── 日志消费协程 ────────────────────────────────────────────

async def _log_consumer():
    """高效消费者：支持日志批处理和状态主动推送"""
    log_accumulator = []
    last_send_time = time.time()

    while True:
        try:
            while not manager.sync_queue.empty():
                item = manager.sync_queue.get_nowait()

                if item["type"] == "state":
                    if log_accumulator:
                        await manager.add_log_batch("".join(log_accumulator))
                        log_accumulator = []
                    await manager.update_state(item["is_running"])

                elif item["type"] == "log":
                    log_accumulator.append(item["content"])
                    if len(log_accumulator) > 50:
                        await manager.add_log_batch("".join(log_accumulator))
                        log_accumulator = []
                        last_send_time = time.time()

            if log_accumulator and (time.time() - last_send_time > 0.05):
                await manager.add_log_batch("".join(log_accumulator))
                log_accumulator = []
                last_send_time = time.time()

        except Exception as e:
            print(f"Log consumer error: {e}")

        await asyncio.sleep(0.01)

# ─── 启动事件 ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时的逻辑 (Startup) ---
    
    # 1. 启动日志消费任务
    asyncio.create_task(_log_consumer())
    
    # 2. 启动打开浏览器的任务
    # 注意：为了不阻塞服务器启动，建议也将打开浏览器的逻辑放入 task 或直接写在这里
    async def open_browser():
        target_url = "http://127.0.0.1:9980"
        await asyncio.sleep(0.5)
        webbrowser.open(target_url)
    
    asyncio.create_task(open_browser())

    yield
    print("退出")
    # --- 关闭时的逻辑 (Shutdown) ---
    # 如果有关闭时需要执行的代码，写在 yield 后面
    pass

# ─── 应用实例 ────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

# ─── 进程管理器（全局单例） ──────────────────────────────────
manager = ProcessManager()

# 将 manager 注入给需要的路由模块
routes_runner.init_manager(manager)

# ─── 注册路由 ────────────────────────────────────────────────
app.include_router(routes_task.router)
app.include_router(routes_runner.router)
app.include_router(routes_media.router)

# ─── 静态文件服务 ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


'''
@app.on_event("startup")
async def on_startup_log_consumer():
    asyncio.create_task(_log_consumer())

@app.on_event("startup")
async def on_startup_open_browser():
    target_url = "http://127.0.0.1:9980"
    await asyncio.sleep(0.5)
    webbrowser.open(target_url)
'''

# ─── WebSocket 日志推送 ──────────────────────────────────────

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    manager.active_websockets.append(websocket)

    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "is_running": manager.is_running,
            "history": "".join(manager.log_buffer)
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in manager.active_websockets:
            manager.active_websockets.remove(websocket)

# ─── 日志 & 控制 API ────────────────────────────────────────

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


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")




# ─── 入口 ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9980)