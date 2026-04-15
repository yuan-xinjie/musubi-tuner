"""
进程管理模块：子进程执行、日志队列、WebSocket 广播
"""
import json
import queue
import subprocess
import threading
from typing import List, Any, Optional

from fastapi import WebSocket, HTTPException


class ProcessManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.sync_queue = queue.Queue()
        self.active_websockets: List[WebSocket] = []
        self.is_running = False
        self.log_buffer: List[str] = []
        self.max_buffer_size = 5000

    async def broadcast(self, data: dict):
        """稳健的 JSON 广播：自动剔除已关闭连接"""
        msg = json.dumps(data)
        to_remove = []
        for ws in list(self.active_websockets):
            try:
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
        self.sync_queue.put({"type": "state", "is_running": True})

        try:
            for cmd in cmds:
                if not self.is_running:
                    break

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0
                )

                def _read_io():
                    try:
                        residual = b""
                        while True:
                            if not self.process or not self.process.stdout:
                                break
                            
                            chunk = self.process.stdout.read(4096)
                            if not chunk:
                                if residual:
                                    self.sync_queue.put({"type": "log", "content": residual.decode('utf-8', errors='replace')})
                                break

                            data = residual + chunk
                            text = ""

                            try:
                                text = data.decode('utf-8')
                                residual = b""
                            except UnicodeDecodeError as e:
                                # 若解码错点在最后几个字节，说明遇到了因分块读取而被腰斩的多字节字符
                                if e.start >= len(data) - 4:
                                    text = data[:e.start].decode('utf-8')
                                    residual = data[e.start:]
                                else:
                                    # 错误并不是在末尾，说明这股输出流根本就不是纯正的 UTF-8 (可能是底层的 GBK 输出)
                                    try:
                                        text = data.decode('gbk')
                                        residual = b""
                                    except UnicodeDecodeError as ge:
                                        if ge.start >= len(data) - 2:
                                            text = data[:ge.start].decode('gbk')
                                            residual = data[ge.start:]
                                        else:
                                            text = data.decode('utf-8', errors='replace')
                                            residual = b""

                            if text:
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
