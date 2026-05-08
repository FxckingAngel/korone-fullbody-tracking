@echo off
cd /d "%~dp0"
title korone-fullbody-tracking

".\.venv\Scripts\python.exe" bin\mediapipepose.py

pause
