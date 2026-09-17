@echo off
rem Fin de semana: entrena sin parar hasta el lunes 07:00
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\FlyOperatorLab\scripts\start_learning.ps1" -UntilTime 07:00 -UntilWeekday Monday
pause
