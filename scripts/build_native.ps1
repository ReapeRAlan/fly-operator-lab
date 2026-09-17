$ErrorActionPreference='Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Push-Location $labRoot
try {
 python scripts/native_profile.py
 New-Item -ItemType Directory -Force -Path work/build | Out-Null
 $vcvars='C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
 $compile='cl /nologo /LD /std:c++17 /O2 /EHsc /MT /Ithird_party /Ithird_party\minhook-1.3.4\include native\bridge.cpp third_party\minhook-1.3.4\src\buffer.c third_party\minhook-1.3.4\src\hook.c third_party\minhook-1.3.4\src\trampoline.c third_party\minhook-1.3.4\src\hde\hde64.c /Fo:work\build\ /link /OUT:work\build\flybridge.dll /DEBUG /PDB:work\build\flybridge.pdb user32.lib gdi32.lib shell32.lib bcrypt.lib opengl32.lib'
 & cmd.exe /d /s /c "`"$vcvars`" >nul && $compile"
 if($LASTEXITCODE -ne 0){throw 'Native compilation failed'}
} finally {Pop-Location}
