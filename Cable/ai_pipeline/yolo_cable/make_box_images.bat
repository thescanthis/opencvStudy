@echo off
cd /d "%~dp0"
python make_box_images.py %*
pause
