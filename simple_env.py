import numpy as np
from task import GalfitTask, Config
from components import *
from copy import deepcopy
from astropy.io import fits
from torchvision import transforms
from torch import from_numpy
import matplotlib.pyplot as plt
import os

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
    error_punish = 1
    mag_maxgap = 5
    channel_num = 2
    state_num = 2
    action_num = 5
    image_size = 256

    def __init__(self, input_file) -> None:
        config = Config(input_file)
        self._task = GalfitTask(config)
        init_file = input_file.replace('.fits','.init')
        if os.path.exists(init_file):
            _chi2,  self._mag_limit, self._base_chi2 = self._task.read_component(init_file)
            # self._base_chi2 = self._chi2
            self._chi2 = self._base_chi2
            self._sky_state = 0 if self._task.components[0].__background__.trainable else 1
            self._current_code = 0
            bulge_radius = 0
            disk_radius = 10000
            for c in self._task.components[1:]:
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
                if c.magnitude > self._mag_limit:  # 限定mag_limit 为 mag_baseline + mag_maxgap, 其中 mag_baseline 为初始 sersic 的 magnitude
                    self._current_code = 0
                    break
                if c.effective_radius < 2 or c.effective_radius > max(self._task.config.image_size) / 2:
                    self._current_code = 0
                    break
                self._current_code += add_code
            if bulge_radius > disk_radius:
                self._current_code = 0
        else:
            self._task.init_guess()
            self._mag_limit = self._task._mag_baseline + self.mag_maxgap
            self._update_state()
            self._base_chi2 = self._chi2

            with open(init_file, 'w') as file:
                print(self._task, file=file)
                print('\n# Mag_limit = '+str(self._mag_limit), file=file)
                print('\n# Base_chi2 = ' + str(self._base_chi2), file = file)

    def _update_state(self):
        self._task.run()
        self._chi2, _Mag_limit, _Base_chi2 = self._task.read_component('./galfit.01')
        os.remove('./galfit.01')
        self._sky_state = 0 if self._task.components[0].__background__.trainable else 1
        self._current_code = 0
        bulge_radius = 0
        disk_radius = 10000
        for c in self._task.components[1:]:
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
            if c.magnitude > self._mag_limit:  # 限定mag_limit 为 mag_baseline + mag_maxgap, 其中 mag_baseline 为初始 sersic 的 magnitude
                self._current_code = 0
                break
            if c.effective_radius < 2 or c.effective_radius > max(self._task.config.image_size) / 2:
                self._current_code = 0
                break
            self._current_code += add_code
        if bulge_radius > disk_radius:
            self._current_code = 0

    def _split(self, target_index):
        min_redius = 1000000
        min_redius_comp = None
        for c in self._task.components[1:]:
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
            return self.current_state, self.reward, True
        elif action == 0:
            self._task.components[0].__background__.trainable ^= 1
        elif action == 1:
            self._split(1)
        elif action == 2:
            self._split(0.5)
        elif action == 3:
            for c in self._task.components[1:]:
                if not c.__sersic_index__.trainable and c.sersic_index == 4:
                    c.set_sersic_index(trainable=True)
        self._update_state()
        return self.current_state, self.reward, self._current_code == 0

    @property
    def reward(self):
        # Base reward, based on chi^2
        # def sigmoid(x):
        #     return 1 / (1 + np.exp(-x))
        # out = sigmoid(10 * np.exp(-5 * self._chi2))
        r = (1 - self._chi2 / self._base_chi2) * self.chi2_weight
        if self._current_code == 0:
            r -= self.error_punish
        return r

    @property
    def current_state(self):
        output_file = self._task.config._output.value
        with fits.open(output_file) as hdus:
            print(len(hdus), output_file)
            residual = np.array(hdus[3].data)
            model = np.array(hdus[2].data)
        image = from_numpy(np.array([residual, model], dtype=np.float64))
        # plt.imshow(np.log(np.abs(image.numpy()[0])))
        # plt.show()
        image = transforms.Resize(self.image_size)(image).numpy()
        # plt.imshow(np.log(np.abs(image[0])))
        # plt.show()
        return np.array([self._current_code, self._sky_state], dtype=np.float64), image
