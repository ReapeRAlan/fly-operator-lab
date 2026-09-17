@echo off
rem Noche: entrena hasta las 07:00 siguientes con la campana de config\learning.json
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\FlyOperatorLab\scripts\start_learning.ps1" -UntilTime 07:00
pause
