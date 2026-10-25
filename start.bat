@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo           RehabVision AI - Local Application Launcher
echo ======================================================================
echo [Notice] Educational Prototype Only. Not clinically validated.
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: 1. Detect Python Executable
set "PYTHON_EXE="

:: Check Python Launcher (py)
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 -c "import torch" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_EXE=py -3"
    )
)

:: Check standard python in PATH
if "!PYTHON_EXE!"=="" (
    where python >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python -c "import torch" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_EXE=python"
        )
    )
)

:: Check Local AppData Python installation
if "!PYTHON_EXE!"=="" (
    for /d %%P in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%P\python.exe" (
            "%%P\python.exe" -c "import torch" >nul 2>&1
            if !ERRORLEVEL! equ 0 (
                set "PYTHON_EXE=%%P\python.exe"
            )
        )
    )
)

if "!PYTHON_EXE!"=="" (
    echo [WARNING] Could not find Python with PyTorch installed. Falling back to 'python'.
    set "PYTHON_EXE=python"
)

echo [OK] Using Python: !PYTHON_EXE!

:: 2. Launch FastAPI Backend in a new window
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
cd /d "%SCRIPT_DIR%backend"
start "RehabVision AI - Backend API" cmd /k "!PYTHON_EXE! -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

:: 3. Launch Vite Frontend in a new window
echo [2/2] Starting Vite Frontend on http://localhost:5173 ...
cd /d "%SCRIPT_DIR%frontend"
start "RehabVision AI - Frontend" cmd /k "npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo ======================================================================
echo Both servers are launching:
echo   - Backend API: http://127.0.0.1:8000/docs
echo   - Frontend App: http://localhost:5173
echo ======================================================================
echo Opening browser in 3 seconds...
timeout /t 3 >nul
start http://localhost:5173

pause
