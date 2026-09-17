import sys,json,hashlib
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lossless_archive import atomic_npz
from learning_checkpoint import load_teaching_examples


def test_npz_preserves_exact_values_layout_types_and_metadata(tmp_path):
    values={'float':np.array([0.,-0.,np.inf,-np.inf,np.nan],np.float32),'ids':np.arange(35,dtype=np.int64).reshape(5,7)[:,::2],
      'column_major':np.asfortranarray(np.arange(30).reshape(5,6)),'empty':np.empty((0,4),bool),'metadata':np.array('{"version":2}')}
    path=tmp_path/'snapshot.npz';atomic_npz(path,values)
    with np.load(path,allow_pickle=False) as z:
        for key,array in values.items():assert z[key].dtype==array.dtype and z[key].shape==array.shape and z[key].tobytes()==array.tobytes()


def test_failed_serialization_preserves_last_good_archive(tmp_path):
    path=tmp_path/'snapshot.npz';atomic_npz(path,{'a':np.array([1,2])});before=path.read_bytes()
    with pytest.raises(ValueError):atomic_npz(path,{'a':np.array([object()],object)})
    assert path.read_bytes()==before and not list(tmp_path.glob('*.tmp'))


def test_teaching_examples_follow_checkpoint_generation_and_verify_hash(tmp_path):
    current=tmp_path/'current.json';directory=tmp_path/'1';directory.mkdir()
    arrays={'features':np.ones((2,4),np.float32),'actions':np.array([1,0]),'masks':np.ones((2,2),bool),'stages':np.array(['move','orient'])}
    path=directory/'examples.npz';atomic_npz(path,arrays)
    current.write_text(json.dumps({'generation':1,'examples_count':2,'examples_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    legacy=tmp_path/'legacy.npz';atomic_npz(legacy,{**arrays,'actions':np.array([0,1])})
    examples=load_teaching_examples(current,legacy);assert [e[1] for e in examples]==[1,0]
    assert [e[3] for e in examples]==['move','orient']
    path.write_bytes(path.read_bytes()+b'altered')
    with pytest.raises(ValueError,match='integrity'):load_teaching_examples(current,legacy)
