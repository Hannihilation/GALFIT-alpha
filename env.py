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
        self._chi2 = self._task.read_component('./galfit.01')
        shutil.rmtree('./galfit.01')

    def do_action(self, action: int):
        change_flag = False
        for comp in self._task.components:
            if action < comp.state_num:
                if comp.state != action:
                    comp.state = action
                    change_flag = True
                action -= comp.state_num
                break
            action -= comp.state_num
        if action >= 0:
            for comp in self._task.components:
                sersic = _split_comp(comp, action % 4)
                if sersic is not None:
                    self._task.add_component(sersic)
                    change_flag = True
                    break
        if change_flag:
            self._update_state()
        return self.current_state, self.reward

    @property
    def reward(self):
        # Base reward, based on chi^2
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        out = sigmoid(10 * np.exp(-5 * self._chi2))
        # Additional reward, based on sersic indices
        count_sersic = 0
        for comp in self._task.components:
            if not isinstance(comp, Sersic):
                continue
            count_sersic += 1
            if comp.state == 0:
                if comp.sersic_index < 0.3 or comp.sersic_index > 7:
                    out -= 0.5 # Sersic Index Penalty
            if comp.amplitude > 14:
                out -= 0.3 # Amplitude Penalty
            if comp.effective_radius < 30 or comp.effective_radius > 400:
                out -= 0.5 # Radius Penalty

    @property
    def current_state(self):
        out = np.ndarray([-1] * 4)
        count_sersic = 0
        for comp in self._task.components:
            if isinstance(comp, Sky):
                out[0] = comp.state
            elif isinstance(comp, Sersic):
                count_sersic += 1
                out[count_sersic] = comp.state
        return out
