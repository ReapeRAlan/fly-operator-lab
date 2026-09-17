param(
  [double]$Hours=24,
  [string]$UntilTime='',
  [ValidateSet('','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')][string]$UntilWeekday='',
  [switch]$ResetDeadline,
  [switch]$NoSupervisor
)
# ASCII only: Windows PowerShell 5.1 reads BOM-less scripts with the ANSI code page.
$ErrorActionPreference='Stop'
$labRoot='D:\FlyOperatorLab'
$python=Join-Path $labRoot '.venv-learning\Scripts\python.exe'
$runtime=Join-Path $labRoot 'work\learning'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
if ($UntilTime) {
  # Deadline = next occurrence of HH:mm (on UntilWeekday when given), from the local clock.
  $parts=$UntilTime.Split(':'); $now=Get-Date
  $target=$now.Date.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
  if ($UntilWeekday) {
    $days=([int][DayOfWeek]$UntilWeekday - [int]$now.DayOfWeek + 7) % 7
    $target=$target.AddDays($days)
    if ($target -le $now) { $target=$target.AddDays(7) }
  } elseif ($target -le $now) { $target=$target.AddDays(1) }
  $Hours=[math]::Round(($target-$now).TotalHours,3)
  if ($Hours -le 0.1) { throw "Deadline $target is already past or too close" }
  $ResetDeadline=$true
  Write-Output ("Training until {0} ({1} h)." -f $target.ToString('yyyy-MM-dd HH:mm'),$Hours)
}
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
Write-Output "Trainer PID $($worker.Id). Dashboard: http://127.0.0.1:8766. An internal lock prevents two concurrent controllers."
