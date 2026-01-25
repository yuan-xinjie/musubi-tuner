@echo off
chcp 65001 >nul 2>&1
:: 切换到当前脚本所在目录，避免因执行路径导致的目录错误
cd /d "%~dp0"

:: 定义核心变量
set VENV_DIR=.venv
set PYTHON_REQUIRED=3.11+

echo ==============================================
echo 检测虚拟环境
echo ==============================================
:: 检测.venv目录是否存在
if exist "%VENV_DIR%" (
    echo [成功] 检测到虚拟环境 %VENV_DIR%，跳过创建流程
    goto ACTIVATE_VENV
) else (
    echo [提示] 未检测到虚拟环境 %VENV_DIR%，开始前置依赖检测...
)

echo ==============================================
echo 检测Python是否安装
echo ==============================================
:: 检测Python是否存在
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo [错误] 未检测到Python环境！
        echo     请先安装Python %PYTHON_REQUIRED% 版本（官网：https://www.python.org/downloads/）
        pause
        exit /b
    ) else (
        set PYTHON_CMD=py
        echo [成功] 检测到Python Launcher (py)
    )
) else (
    set PYTHON_CMD=python
    echo [成功] 检测到Python环境
)

echo ==============================================
echo 检测UV是否安装
echo ==============================================
:: 检测uv是否已安装（全局可调用）
uv --version >nul 2>&1
if errorlevel 1 (
    echo [提示] 未检测到UV工具，开始自动安装...
    :: Windows优先用pip安装uv
    %PYTHON_CMD% -m pip install uv --upgrade >nul 2>&1
    if errorlevel 1 (
        echo [错误] pip安装UV失败！请手动执行：pip install uv
        pause
        exit /b
    )
    echo [成功] UV工具安装成功
) else (
    echo [成功] 检测到UV工具已安装
)

echo ==============================================
echo UV创建虚拟环境
echo ==============================================
:: 用UV创建虚拟环境
uv venv -p 3.11 --seed "%VENV_DIR%"
if errorlevel 1 (
    echo [错误] UV创建虚拟环境失败！
    echo     可能原因：Python版本低于3.11 / UV安装异常
    pause
    exit /b
)
echo [成功] 虚拟环境 %VENV_DIR% 创建成功

:ACTIVATE_VENV
echo ==============================================
echo 激活虚拟环境并安装依赖
echo ==============================================
:: 激活虚拟环境
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
    echo [错误] 无法激活虚拟环境，请检查 .venv 文件夹是否存在。
    pause
    exit /b
)

echo 正在安装 PyTorch...
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

echo 正在安装当前项目(Editable模式)...
pip install -e .

echo 正在安装其他依赖...
pip install ascii-magic matplotlib tensorboard prompt-toolkit
pip install tomlkit fastapi pydantic typing queue
pip install -U "triton-windows<3.6"

echo 依赖安装完成
echo ==============================================
echo [完成] 所有流程执行完毕！
echo ==============================================
pause