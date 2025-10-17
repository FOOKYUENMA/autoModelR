@echo off
REM Build a standalone Windows executable for AutoModelR
REM This script creates a virtual environment, installs dependencies,
REM and runs PyInstaller to produce AutoModelR.exe in the dist folder.

SET VENV_DIR=.venv
IF NOT EXIST %VENV_DIR% (
    python -m venv %VENV_DIR%
)
CALL %VENV_DIR%\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
REM Build the executable
pyinstaller --onefile app.py --name AutoModelR
IF EXIST dist\AutoModelR.exe (
    echo Build succeeded. Executable is in the dist folder.
) ELSE (
    echo Build failed. Please check the output above for errors.
)