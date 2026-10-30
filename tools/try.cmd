@echo off
rem Drag one or more bead pattern images onto this file to try recognition.
rem The result page opens in the browser (build\try\TIME\index.html).
cd /d "%~dp0.."
set PYTHONUTF8=1
".venv\Scripts\python.exe" tools\try_recognize.py %*
if errorlevel 1 pause
