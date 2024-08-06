from weakref import WeakValueDictionary
from typing import Union
import numpy as np
from task import GalfitTask, Config
from components import *
from copy import deepcopy
import shutil


def _split_comp(targ_sersic: Sersic, fixed_index: float = None):
    if not isinstance(targ_sersic, Sersic):
        return None
    out_sersic = deepcopy(targ_sersic)
    targ_sersic.magnitude += 0.75
    out_sersic.magnitude += 0.75
    # out_sersic.effective_radius *= 2
    if fixed_index is not None:
        out_sersic.set_sersic_index(fixed_index, False)
    else:
        out_sersic.set_sersic_index(1, True)
    return out_sersic


class GalfitEnv:
    def __init__(self, input_file) -> None:
        config = Config(input_file)
        self._task = GalfitTask(config)
        self._task.init_guess()
        self._update_state()

    def _update_state(self):
        self._task.run()
        self.chi2 = self._task.read_component('./galfit.01')
        shutil.rmtree('./galfit.01')

    def do_action(self, action: int):
        change_flag = False
        for comp in self._task.components:
            if action < comp.state_num:
                if comp.state != action:
                    comp.state = action
                    action -= comp.state_num
                    change_flag = True
                break
            action -= comp.state_num
        if action < 3 and action >= 0:
            sersic = _split_comp(self._task.components[action])

    @property
    def current_state(self):
        pass
