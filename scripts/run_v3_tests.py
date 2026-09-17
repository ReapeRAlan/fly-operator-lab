"""Run the project suite and publish a compact, machine-readable validation record."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
command=[sys.executable,'-m','pytest','-q']
completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,encoding='utf-8',errors='replace')
output=(completed.stdout or '')+(completed.stderr or '')
passed=re.search(r'(\d+) passed',output);warnings=re.search(r'(\d+) warnings?',output);seconds=re.search(r'in ([0-9.]+)s',output)
record={'version':'3.1','passed':completed.returncode==0,'tests':int(passed.group(1)) if passed else None,
        'warnings':int(warnings.group(1)) if warnings else 0,'seconds':float(seconds.group(1)) if seconds else None,
        'command':' '.join(command[1:]),'timestamp':datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'summary':output.strip().splitlines()[-1] if output.strip() else ''}
path=ROOT/'outputs/v3_test_validation.json';path.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
if completed.returncode:
    print(output);raise SystemExit(completed.returncode)
