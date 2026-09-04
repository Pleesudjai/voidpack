@echo off
REM VoidPack local launcher. Starts the bridge on the real produce scan and opens the page.
REM Needs OPENAI_API_KEY in the environment and the ASU VPN at sslvpn.asu.edu/2fa.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found on PATH. Install Python 3.11 or newer and retry.
    pause
    exit /b 1
)

if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY is not set. Get a key from https://voyager.rc.asu.edu then run
    echo   setx OPENAI_API_KEY your_key_here
    echo and open a NEW terminal before running this again.
    pause
    exit /b 1
)

echo Installing requirements, quiet, skipped when already satisfied ...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo pip install failed. See the message above.
    pause
    exit /b 1
)

echo Checking the AIR endpoint, needs the ASU VPN ...
curl -s -m 5 -o nul https://openai.rc.asu.edu/v1/models
if errorlevel 1 (
    echo WARNING openai.rc.asu.edu is unreachable. Connect the ASU VPN or every rule will fail.
    echo The page will still open and the solver still packs, only the model calls fail.
)

echo Starting the bridge on scans\real_produce.npy, the initial pack takes about 20 s ...
start "" /b cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:8000"
python -m src.air.bridge --scan scans\real_produce.npy
endlocal
