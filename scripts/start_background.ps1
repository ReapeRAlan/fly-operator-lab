$ErrorActionPreference='Stop'
$labRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $labRoot '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $taskPython)){$taskPython=(Get-Command python).Source}
Start-Process -FilePath $taskPython -ArgumentList @('-u','scripts/run_local.py','--episodes','20','--seconds','3') -WorkingDirectory $labRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $labRoot 'work/user_run.log') -RedirectStandardError (Join-Path $labRoot 'work/user_run_error.log')
