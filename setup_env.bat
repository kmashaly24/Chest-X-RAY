@echo off
REM Create the conda environment and install all dependencies.
call conda create -n chest_xray python=3.11 pip -y
call conda run -n chest_xray python -m pip install --upgrade pip
call conda run -n chest_xray pip install -r requirements.txt
echo.
echo Environment 'chest_xray' is ready.
echo Activate it with: conda activate chest_xray
pause