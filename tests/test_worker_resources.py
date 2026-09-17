import sys
from pathlib import Path
import pytest
import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from learning_runtime import configure_worker_process

class Process:
    def __init__(self):self.affinity=list(range(20));self.priority=None
    def cpu_affinity(self,value=None):
        if value is not None:self.affinity=value
        return self.affinity
    def nice(self,value):self.priority=value

def test_explicit_fast_cores_and_legacy_quiet_profile():
    process=Process()
    configure_worker_process(process,{'worker_cpu_affinity':[8,2,4,6],'worker_priority':'below_normal','max_cpu_threads':2})
    assert process.affinity==[2,4,6,8]
    assert process.priority==psutil.BELOW_NORMAL_PRIORITY_CLASS
    process=Process()
    configure_worker_process(process,{'max_cpu_threads':1,'worker_priority':'idle'})
    assert process.affinity==[19] and process.priority==psutil.IDLE_PRIORITY_CLASS

@pytest.mark.parametrize('affinity',[[],[20],[-1],[True],'2,4'])
def test_invalid_affinity_fails_before_mutation(affinity):
    process=Process()
    with pytest.raises(ValueError):configure_worker_process(process,{'worker_cpu_affinity':affinity})
    assert process.affinity==list(range(20)) and process.priority is None
