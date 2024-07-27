from components import *
from astropy.io import fits
import subprocess


class Config:
    def __init__(self, input_file, output_file=None, psf_file='none', sigma_file='none', mask_file='none'):
        self._input = StrParam('A', input_file)
        if output_file is None:
            output_file = input_file.replace('.fit', '_out.fits')
        self._output = StrParam('B', output_file)
        self._psf = StrParam('D', psf_file)
        self._sigma = StrParam('C', sigma_file)
        self._mask = StrParam('F', mask_file)
        self._mode = StrParam('P', 0)
        input_file = fits.open(self._input.value)
        psf_file = fits.open(self._psf.value)
        scale = self._read_header(psf_file[0], 'SCALE')
        self._psf_scale = StrParam('E', scale)
        self._constrains = StrParam('G', 'none')
        in_s1 = self._read_header(input_file[0], 'NAXIS1')
        in_s2 = self._read_header(input_file[0], 'NAXIS2')
        self._image_region = StrParam('H', f"1 {in_s1} 1 {in_s2}")
        psf_s1 = self._read_header(psf_file[0], 'NAXIS1')
        psf_s2 = self._read_header(psf_file[0], 'NAXIS2')
        self._convolution_size = StrParam('I', f"{psf_s1} {psf_s2}")
        zp = self._read_header(input_file[0], 'ZPT_GSC')
        self._zeropoint = StrParam('J', zp)
        cd11 = self._read_header(input_file[0], 'CD1_1')
        cd12 = self._read_header(input_file[0], 'CD1_2')
        cd21 = self._read_header(input_file[0], 'CD2_1')
        cd22 = self._read_header(input_file[0], 'CD2_2')
        dx = np.sqrt(cd11**2+cd12**2)
        dy = np.sqrt(cd21**2+cd22**2)
        self._pixel_scale = StrParam('K', f"{dx} {dy}")
        self._display_type = StrParam('O', 'regular')
        input_file.close()
        psf_file.close()
        self.parameters = [self._input, self._output, self._sigma, self._psf,
                           self._psf_scale, self._mask, self._constrains, self._image_region,
                           self._convolution_size, self._zeropoint, self._pixel_scale, self._display_type, self._mode]

    def _read_header(self, hdu, key):
        if key in hdu.header:
            return hdu.header[key]
        else:
            return hdu.header['_'+key[1:]]

    @property
    def galfit_mode(self):
        return self._mode.value

    @galfit_mode.setter
    def galfit_mode(self, mode):
        self._mode.value = mode

    @property
    def pixel_scale(self):
        value = re.split(r'\s+', self._pixel_scale.value)
        dx, dy = float(value[0]), float(value[1])
        return np.sqrt(dx**2+dy**2) * 3600

    @property
    def zeropoint(self):
        value = self._zeropoint.value
        if isinstance(value, str):
            value = re.split(r'\s+', value)
            return float(value[0])
        return value

    def __repr__(self) -> str:
        s = ''
        for param in self.parameters:
            s += param.__repr__() + '\n'
        return s


class GalfitTask:
    def __init__(self, config):
        self._config = config
        self._components = []

    @property
    def config(self):
        return self._config

    @property
    def components(self):
        return self._components

    def add_component(self, component: Component):
        self._components.append(component)

    def remove_component(self, index=-1):
        self._components.pop(index)

    def __repr__(self) -> str:
        s = self._config.__repr__() + '\n'
        for component in self._components:
            s += component.__repr__() + '\n'
        return s

    def read_component(self, file_name):
        self._components = []
        with open(file_name, 'r') as file:
            line = file.readline()
            while line:
                line = line.lstrip()
                pos = line.find(')')
                if len(line) > 0 and pos > 0:
                    if line[0] == '0':
                        line = line.split(' ')
                        component = component_names[line[1]]()
                        file = component.read(file)
                        self._components.append(component)
                line = file.readline()

    def run(self, galfit_file=None, galfit_mode=0):
        if galfit_file is None:
            galfit_file = self._config.__output__.value.replace(
                '.fits', '.galfit')
        self.config.galfit_mode = galfit_mode
        with open(galfit_file, 'w') as file:
            print(self, file=file)
        subprocess.run(['./galfit', galfit_file], check=True)
