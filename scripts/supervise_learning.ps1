param([int]$MaxRestarts=6,[int]$IntervalSeconds=60,[double]$ForgiveAfterHours=3)
# Relaunch the trainer if it exits unexpectedly before the campaign deadline.
# Intentional ends (panel stop, pilot finished, curriculum complete, calibration failure) are respected.
# Restarts never reset the persisted deadline, and learning resumes from the last checkpoint.
# After ForgiveAfterHours without a crash the restart budget is restored (long weekend campaigns).
$ErrorActionPreference='Continue'
$labRoot='D:\FlyOperatorLab'
$runtime=Join-Path $labRoot 'work\learning'
$log=Join-Path $runtime 'supervisor.log'
function Write-Log($text) { "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$text | Add-Content -LiteralPath $log -Encoding utf8 }
$cfg=Get-Content -LiteralPath (Join-Path $labRoot 'config\learning.json') -Raw | ConvertFrom-Json
$schedulePath=Join-Path $runtime ("campaigns\{0}\schedule.json" -f $cfg.experiment_id)
$restarts=0
Write-Log "supervisor started for $($cfg.experiment_id) (pid $PID)"
while ($true) {
  Start-Sleep -Seconds $IntervalSeconds
  try { $schedule=Get-Content -LiteralPath $schedulePath -Raw | ConvertFrom-Json } catch { Write-Log "schedule unreadable: $_"; continue }
  if (-not $schedule.pilot_deadline) { continue }
  $deadline=[DateTimeOffset]::FromUnixTimeMilliseconds([long]([double]$schedule.pilot_deadline*1000)).LocalDateTime
  if ((Get-Date) -ge $deadline) { Write-Log "deadline $deadline reached"; break }
  $alive=Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train_curriculum' }
  if ($alive) {
    if ($restarts -gt 0 -and $lastRestart -and ((Get-Date)-$lastRestart).TotalHours -ge $ForgiveAfterHours) { Write-Log "stable for $ForgiveAfterHours h; restart budget restored"; $restarts=0 }
    continue
  }
  try { $status=Get-Content -LiteralPath (Join-Path $runtime 'status.json') -Raw | ConvertFrom-Json } catch { $status=$null }
  if ($status -and $status.run -eq $cfg.experiment_id -and $status.state -in @('stopped','curriculum_complete','needs_calibration')) {
    Write-Log "trainer ended intentionally: state=$($status.state) reason=$($status.reason)"; break
  }
  if ($restarts -ge $MaxRestarts) { Write-Log "restart limit $MaxRestarts reached; last state=$($status.state)"; break }
  $stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
  foreach ($name in 'worker.stderr.log','worker.stdout.log') {
    $path=Join-Path $runtime $name
    if ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -gt 0) { Copy-Item -LiteralPath $path -Destination (Join-Path $runtime "crash_${stamp}_$name") }
  }
  $restarts++; $lastRestart=Get-Date
  Write-Log "trainer not running (last state=$($status.state), reason=$($status.reason)); restart $restarts of $MaxRestarts"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $labRoot 'scripts\start_learning.ps1') -Hours 1 -NoSupervisor *>> $log
}
Write-Log 'supervisor exiting'
