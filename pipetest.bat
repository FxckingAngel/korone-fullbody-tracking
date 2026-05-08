@echo off
cd /d "%~dp0"
title korone-fullbody-tracking - Pipe Test

".\.venv\Scripts\python.exe" bin\tests\pipetest.py

pause
