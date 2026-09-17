param([double]$Hours=24,[switch]$ResetDeadline,[switch]$NoSupervisor)
$ErrorActionPreference='Stop'
$labRoot='D:\FlyOperatorLab'
$python=Join-Path $labRoot '.venv-learning\Scripts\python.exe'
$runtime=Join-Path $labRoot 'work\learning'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
try { $null=Invoke-RestMethod 'http://127.0.0.1:8766/api/state' -TimeoutSec 2 }
catch {
  $server=Start-Process $python -ArgumentList (Join-Path $labRoot 'scripts\serve_lab.py') -WorkingDirectory $labRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'server.stdout.log') -RedirectStandardError (Join-Path $runtime 'server.stderr.log') -PassThru
  $server.Id | Set-Content (Join-Path $runtime 'server.pid')
}
$sessionPath=Join-Path $labRoot 'work\session.json'
$running=$false
if (Test-Path -LiteralPath $sessionPath) {
  $session=Get-Content -LiteralPath $sessionPath | ConvertFrom-Json
  $game=Get-Process -Id $session.pid -ErrorAction SilentlyContinue
  $running=$game -and $game.Path -eq 'D:\SteamLibrary\steamapps\common\DoorKickers2\DoorKickers2.exe'
}
if (!$running) { & $python (Join-Path $labRoot 'scripts\launch.py'); if ($LASTEXITCODE -ne 0) {throw 'Game launch failed'} }
$workerArgs=@((Join-Path $labRoot 'scripts\train_curriculum.py'),'--hours',$Hours.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($ResetDeadline) { $workerArgs+='--reset-deadline' }
$worker=Start-Process $python -ArgumentList $workerArgs -WorkingDirectory $labRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'worker.stdout.log') -RedirectStandardError (Join-Path $runtime 'worker.stderr.log') -PassThru
$worker.Id | Set-Content (Join-Path $runtime 'worker.pid')
if (-not $NoSupervisor) {
  $supervising=Get-CimInstance Win32_Process -Filter "Name like 'powershell%'" | Where-Object { $_.CommandLine -match 'supervise_learning' }
  if (-not $supervising) { Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $labRoot 'scripts\supervise_learning.ps1')) -WindowStyle Hidden | Out-Null }
}
Write-Output "Entrenador PID $($worker.Id). Panel: http://127.0.0.1:8766. El bloqueo interno impide dos controladores simultáneos."
