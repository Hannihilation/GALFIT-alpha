import numpy as np
from task import GalfitTask, Config
from components import *
import shutil
from copy import deepcopy
from astropy.io import fits

# code_state = ['bulge', 'disk', 'bar', 'bulge&disk',
#               'disk&bar', 'bulge&bar', 'bulge&disk&bar', 'error']
code_state = ['error', 'bulge', 'disk', 'bulge&disk',
              'bar', 'bulge&bar', 'disk&bar', 'bulge&disk&bar']

state_code = {}
for i, s in enumerate(code_state):
    state_code[s] = i

# free or 4: bulge, 0.5: bar, 1: disk


class GalfitEnv:
    chi2_weight = 20

    def __init__(self, input_file) -> None:
        config = Config(input_file)
        self._task = GalfitTask(config)
        self._task.init_guess()
        self._update_state()

    def _update_state(self):
        self._task.run()
        self._chi2 = self._task.read_component('./galfit.01')
        shutil.rmtree('./galfit.01')
        self._sky_state = 0 if self._task.component[0].__background__.trainable else 1
        self._current_code = 0
        bulge_radius = 0
        disk_radius = 10000
        for c in self._task.component[1:]:
            if c.__sersic_index__.trainable:
                if c.sersic_index > 7 or c.sersic_index < 2.5:
                    self._current_code = 0
                    break
                add_code = 1
                bulge_radius = c.effective_radius
            elif c.sersic_index == 4:
                add_code = 1
                bulge_radius = c.effective_radius
            elif c.sersic_index == 1:
                add_code = 2
                disk_radius = c.effective_radius
            elif c.sersic_index == 0.5:
                add_code = 4
            if self._current_code & add_code:
                self._current_code = 0
                break
            if c.amplitude > 14:
                r -= 0.3  # Amplitude Penalty
            if c.effective_radius < 2 or c.effective_radius > max(self._task.config.image_size) / 2:
                r -= 0.5  # Radius Penalty
            self._current_code += add_code
        if bulge_radius > disk_radius:
            self._current_code = 0

    def _split(self, target_index):
        min_redius = 1000000
        min_redius_comp = None
        for c in self._task.component[1:]:
            if c.effective_radius < min_redius:
                min_redius = c.effective_radius
                min_redius_comp = c
        add_comp = deepcopy(min_redius_comp)
        min_redius_comp.magnitude += 0.75
        add_comp.magnitude += 0.75
        min_redius_comp.set_sersic_index(4, True)
        add_comp.set_sersic_index(target_index, False)
        add_comp.effective_radius *= 2
        self._task.add_component(add_comp)

    def step(self, action: int):
        """
        :param action: int, switch_sky: 0, split_disk: 1, split_bar: 2, free_bulge: 3, stop: 4

        :returns (current_state, reward, done)
        """
        if action == 4:
            return 0, 0, True
        elif action == 0:
            self._task.component[0].__background__.trainable ^= 1
        elif action == 1:
            self._split(1)
        elif action == 2:
            self._split(0.5)
        elif action == 3:
            for c in self._task.component[1:]:
                if not c.__sersic_index__.trainable and c.sersic_index == 4:
                    c.set_sersic_index(trainable=True)
        self._update_state()
        return self.current_state, self.reward, False

    @property
    def reward(self):
        # Base reward, based on chi^2
        # def sigmoid(x):
        #     return 1 / (1 + np.exp(-x))
        # out = sigmoid(10 * np.exp(-5 * self._chi2))
        r = (1 - self._chi2 / self._base_chi2) * self.chi2_weight
        if self._current_code == 0:
            r -= 1
        return r

    @property
    def current_state(self):
        """
        return state: [code, sky, residual, model]
        """
        output_file = self._task.config.__output__.value
        with fits.open(output_file) as hdus:
            residual = hdus[3].data
            model = hdus[2].data
        return [self._current_code, self._sky_state, residual, model]
