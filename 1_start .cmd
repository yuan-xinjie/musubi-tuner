@echo off
:: 切换到当前脚本所在的目录，防止因为右键管理员运行导致路径错误
cd /d "%~dp0"

echo 正在激活虚拟环境...
:: 关键步骤：使用 call 来调用 activate，否则脚本会在这里结束
call .\.venv\Scripts\activate

:: 检查激活是否成功（可选）
if errorlevel 1 (
    echo [错误] 无法激活虚拟环境，请检查 .venv 文件夹是否存在。
    pause
    exit /b
)

echo 启动主程序...
python .\src\ui.py

pause