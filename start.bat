@echo off
title Defence CV App
cd /d "%~dp0"

echo Starting Defence CV App...
echo.

:: Kill any old instance on port 8383
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8383 "') do taskkill /PID %%a /F >nul 2>&1

:: Start combined server (static files + inference API) in a visible window
start "Defence CV Server (port 8383)" cmd /k "py server.py"

:: Wait for it to load the model (~5s) then open browser
timeout /t 6 /nobreak >nul
start "" "http://127.0.0.1:8383"

echo Server started at http://127.0.0.1:8383
echo Keep the server window open while using the app.
