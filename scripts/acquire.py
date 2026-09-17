"""Acquire version-pinned public inputs; verify upstream hashes, retain provenance."""
from pathlib import Path
import requests, hashlib, base64, json, time, zipfile, io
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
DATA = ['body-annotations-male-cns-v1.0-minconf-0.5.feather',
        'body-neurotransmitters-male-cns-v1.0.feather',
        'connectome-weights-male-cns-v1.0-minconf-0.5.feather']

def acquire(url, dest, sha256=None):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers['Accept-Encoding']='identity'
    h = s.head(url, allow_redirects=True, timeout=60); h.raise_for_status()
    size = int(h.headers.get('Content-Length', 0))
    md5_expected = next((v.strip()[4:] for v in h.headers.get('x-goog-hash','').split(',') if v.strip().startswith('md5=')), None)
    if md5_expected: md5_expected = md5_expected.strip()
    if h.headers.get('Content-Encoding'): size=0
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix+'.partial')
        with s.get(url, stream=True, timeout=(30,120)) as r:
            r.raise_for_status(); last=time.monotonic(); n=0
            with tmp.open('wb') as f:
                for chunk in r.iter_content(4*1024*1024):
                    f.write(chunk); n+=len(chunk)
                    if time.monotonic()-last > 10:
                        print(f'{dest.name}: {n/1e6:.0f}/{size/1e6:.0f} MB',flush=True); last=time.monotonic()
        if size and tmp.stat().st_size!=size: raise ValueError(f'Size mismatch: {tmp}')
        tmp.replace(dest)
    md5=hashlib.md5(); sha=hashlib.sha256()
    with dest.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): md5.update(b); sha.update(b)
    if md5_expected and base64.b64encode(md5.digest()).decode()!=md5_expected: raise ValueError(f'MD5 mismatch: {dest}')
    if sha256 and sha.hexdigest()!=sha256: raise ValueError(f'SHA256 mismatch: {dest}')
    record={'url':url,'path':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':sha.hexdigest(),'upstream_md5':md5_expected,'verified_at_unix':time.time()}
    dest.with_suffix(dest.suffix+'.provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('VERIFIED',dest.name,record['bytes'],flush=True)
    return dest

def dataset():
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda n:acquire(BASE+n,ROOT/'data/raw'/n),DATA))

def toolchain():
    gh = requests.get('https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/tags/Ghidra_12.1.3_build',timeout=30).json()
    asset = next(a for a in gh['assets'] if a['name'].endswith('.zip'))
    digest=asset.get('digest','').removeprefix('sha256:') or None
    z=acquire(asset['browser_download_url'],ROOT/'third_party'/asset['name'],digest)
    out=ROOT/'third_party/ghidra_12.1.3_PUBLIC'
    if not out.exists():
        with zipfile.ZipFile(z) as zz: zz.extractall(ROOT/'third_party')
    jd=requests.get('https://api.adoptium.net/v3/assets/latest/25/hotspot',params={'architecture':'x64','image_type':'jdk','os':'windows'},timeout=30).json()[0]
    pkg=jd['binary']['package']; z=acquire(pkg['link'],ROOT/'third_party'/pkg['name'],pkg['checksum'])
    with zipfile.ZipFile(z) as zz:
        top=zz.namelist()[0].split('/')[0]
        if not (ROOT/'third_party'/top).exists(): zz.extractall(ROOT/'third_party')
    (ROOT/'config/java_home.txt').write_text(str(ROOT/'third_party'/top))
    z=acquire('https://github.com/TsudaKageyu/minhook/archive/refs/tags/v1.3.4.zip',ROOT/'third_party/minhook-v1.3.4.zip')
    if not (ROOT/'third_party/minhook-1.3.4').exists():
        with zipfile.ZipFile(z) as zz: zz.extractall(ROOT/'third_party')
    acquire('https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp',ROOT/'third_party/json.hpp')
    acquire('https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h',ROOT/'third_party/stb_image_write.h','cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05')
    acquire('https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/model.py',ROOT/'third_party/shiu_reference_model.py')
    acquire('https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/LICENSE',ROOT/'third_party/shiu_LICENSE')

if __name__=='__main__':
    import sys
    {'dataset':dataset,'toolchain':toolchain}[sys.argv[1]]()
