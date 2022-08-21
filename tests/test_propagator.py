from mohou.propagator import TerminateChecker
from mohou.types import TerminateFlag


def test_terminate_checker():

    is_termianted = TerminateChecker(n_check_flag=3, terminate_threshold=0.95)

    assert not is_termianted([TerminateFlag.from_bool(b) for b in [True, True]])
    assert not is_termianted([TerminateFlag.from_bool(b) for b in [False, False, False]])
    assert is_termianted([TerminateFlag.from_bool(b) for b in [True, True, True]])
    assert is_termianted([TerminateFlag.from_bool(b) for b in [False, True, True, True]])
