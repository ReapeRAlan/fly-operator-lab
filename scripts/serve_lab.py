from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import uvicorn
if __name__=='__main__':uvicorn.run('lab_api:app',host='127.0.0.1',port=8766,access_log=False)
