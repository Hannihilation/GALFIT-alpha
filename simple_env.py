import numpy as np
from task import GalfitTask, Config
from components import *
from copy import deepcopy
from astropy.io import fits
from torchvision import transforms
from torch import from_numpy
import matplotlib.pyplot as plt
import os
import shutil
from astropy.stats import sigma_clipped_stats

# code_state = ['bulge', 'disk', 'bar', 'bulge&disk',
#               'disk&bar', 'bulge&bar', 'bulge&disk&bar', 'error']
code_state = ['error', 'bulge', 'bulge&disk', 'bulge&disk&bar']

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
    action_num = 4
    image_size = 256

    def __init__(self, config: Config) -> None:
        self._task = GalfitTask(config)
        init_file = self._task.config.input_file.replace('.fits', '.init')
        self._output_file = self._task.config.output_file.replace(
            '.fits', '.save')
        self._mask_file = config.mask_file
        if os.path.exists(init_file) and os.path.exists(self._output_file):
            self._chi2 = self._task.read_component(init_file)
        else:
            self._task.init_guess()
            if os.path.exists('./galfit.01'):
                os.remove('./galfit.01')
            crash = self._task.run()
            if crash:
                raise ValueError('Galfit crashed at init guess!')
            self._chi2 = self._task.read_component('./galfit.01')
            shutil.move(self._task.config._output.value, self._output_file)
            with open(init_file, 'w') as file:
                print(f'#  Chi^2/nu = {self._chi2}', file=file)
                print(self._task, file=file)
        self._sky_state = 0 if self._task.components[0].__background__.trainable else 1
        self._current_code = -1
        self._base_chi2 = self.chi2
        self._mag_limit = self._task.components[1].magnitude + \
            self.mag_maxgap
        size = self._task.config.image_size
        x, y = np.meshgrid(np.linspace(-size/2, size/2, size), np.linspace(-size/2, size/2, size))
        half_weight_size = (config.galaxy_range[1] - config.galaxy_range[0]) / 5
        self._image_weight = np.exp2(-(x**2+y**2)/half_weight_size**2)

        with fits.open(self._task.config.input_file) as hdul:
            data = np.array(hdul[0].data)
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        total_noise = np.sqrt(np.abs(data) + sky_std **2)
        sigmap = total_noise
        self._sigma_image = sigmap

    def _update_state(self):
        if os.path.exists('./galfit.01'):
            os.remove('./galfit.01')
        crash = self._task.run()
        self._sky_state = 0 if self._task.components[0].__background__.trainable else 1
        if crash:
            self._current_code = 0
            return
        self._chi2 = self._task.read_component('./galfit.01')
        self._current_code = -1
        self._output_file = self._task.config._output.value

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
        min_redius_comp.set_sersic_index(4, False)
        add_comp.set_sersic_index(target_index, False)
        add_comp.effective_radius *= 2
        self._task.add_component(add_comp)

    def step(self, action: int):
        """
        :param action: int, switch_sky: 0, split: 1, free_bulge: 2, stop: 3

        :returns (current_state, reward, done)
        """
        change = True
        if action == 3:
            return self.current_state, self.reward, True
        elif action == 0:
            self._task.components[0].__background__.trainable ^= 1
        elif action == 1:
            if self.current_code == state_code['bulge']:
                self._split(1)
            elif self.current_code == state_code['bulge&disk']:
                self._split(0.5)
            else:
                change = False
        elif action == 2:
            for c in self._task.components[1:]:
                if not c.__sersic_index__.trainable:
                    c.set_sersic_index(4, True)
                else:
                    change = False
        if change:
            self._update_state()
        return self.current_state, self.reward, self.current_code == 0

    @property
    def current_code(self):
        if self._current_code < 0:
            self._current_code = 0
            bulge_radius = 0
            disk_radius = 10000
            for c in self._task.components[1:]:
                self._current_code += 1
                if c.__sersic_index__.trainable:
                    if c.sersic_index > 7 or c.sersic_index < 2.5:
                        self._current_code = 0
                        break
                    bulge_radius = c.effective_radius
                elif c.sersic_index == 4:
                    bulge_radius = c.effective_radius
                elif c.sersic_index == 1:
                    disk_radius = c.effective_radius
                if c.magnitude > self._mag_limit:  # 限定mag_limit 为 mag_baseline + mag_maxgap, 其中 mag_baseline 为初始 sersic 的 magnitude
                    self._current_code = 0
                    break
                if c.effective_radius < 2 or c.effective_radius > max(self._task.config.image_size) / 2:
                    self._current_code = 0
                    break
            if bulge_radius > disk_radius:
                self._current_code = 0
        return self._current_code

    @property
    def chi2(self):
        # return self._chi2
        with fits.open(self._output_file) as hdus:
            residual = np.array(hdus[3].data)
        with fits.open(self._task.config.mask_file) as mask:
            mask_data = mask[0].data

        chi2 = np.sum(self._image_weight* (1 - mask_data) *np.abs(residual) / self._sigma_image)
        return chi2

    @property
    def reward(self):
        # Base reward, based on chi^2
        # def sigmoid(x):
        #     return 1 / (1 + np.exp(-x))
        # out = sigmoid(10 * np.exp(-5 * self._chi2))
        tmp = self.chi2
        r = (1 - tmp / self._base_chi2) * self.chi2_weight
        if self.current_code == 0:
            r -= self.error_punish
        self._base_chi2 = tmp
        return r

    @property
    def current_state(self):
        with fits.open(self._output_file) as hdus:
            residual = np.array(hdus[3].data)
            model = np.array(hdus[2].data)
        with fits.open(self._mask_file) as mask:
            residual = residual * (1 - np.array(mask[0].data))
        image = from_numpy(np.array([residual, model], dtype=np.float64))
        # plt.imshow(np.log(np.abs(image.numpy()[0])))
        # plt.show()
        (l_margin, r_margin) = self._galaxy_range
        image = transforms.Resize(self.image_size)(image[l_margin:r_margin, l_margin:r_margin]).numpy()
        # plt.imshow(np.log(np.abs(image[0])))
        # plt.show()
        return np.array([self.current_code, self._sky_state], dtype=np.float64), image
