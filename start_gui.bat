@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Python-Umgebung fehlt. Bitte README lesen.
  pause
  exit /b 1
)
start "" .venv\Scripts\pythonw.exe -m map2mc gui
