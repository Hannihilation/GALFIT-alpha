from components import *
from astropy.io import fits
import subprocess
import os
from photutils.segmentation import SourceFinder, SourceCatalog
from photutils.background import Background2D, MedianBackground


def _read_header(hdu, key, default=None):
    if key in hdu.header:
        return hdu.header[key]
    elif '_'+key[1:] in hdu.header:
        return hdu.header['_'+key[1:]]
    if default is None:
        raise KeyError(f'Keyword {key} not found in header')
    return default


class Config:
    def __init__(self, input_file, output_file=None, psf_file=None, sigma_file='none', mask_file=None):
        self._input = StrParam('A', input_file)
        if output_file is None:
            output_file = input_file.replace('.fits', '_out.fits')
        self._output = StrParam('B', output_file)
        if psf_file == None:
            psf_file = input_file.replace('.fits', '_ep.fits')
        self._psf = StrParam('D', psf_file)
        self._sigma = StrParam('C', sigma_file)
        if mask_file is None:
            mask_file = input_file.replace('.fits', '_mm.fits')
        self._mask = StrParam('F', mask_file)
        self._mode = StrParam('P', 0)
        input_file = fits.open(self._input.value)
        psf_file = fits.open(self._psf.value)
        scale = _read_header(psf_file[0], 'SCALE')
        self._psf_scale = StrParam('E', scale)
        self._constrains = StrParam('G', 'none')
        in_s1 = _read_header(input_file[0], 'NAXIS1')
        in_s2 = _read_header(input_file[0], 'NAXIS2')
        self._image_region = StrParam('H', f"1 {in_s1} 1 {in_s2}")
        psf_s1 = _read_header(psf_file[0], 'NAXIS1')
        psf_s2 = _read_header(psf_file[0], 'NAXIS2')
        self._convolution_size = StrParam('I', f"{psf_s1} {psf_s2}")
        zp = _read_header(input_file[0], 'ZPT_GSC')
        self._zeropoint = StrParam('J', zp)
        cd11 = _read_header(input_file[0], 'CD1_1')
        cd12 = _read_header(input_file[0], 'CD1_2', default=0)
        cd21 = _read_header(input_file[0], 'CD2_1', default=0)
        cd22 = _read_header(input_file[0], 'CD2_2')
        dx = np.sqrt(cd11**2+cd12**2)
        dy = np.sqrt(cd21**2+cd22**2)
        self._pixel_scale = StrParam('K', f"{dx} {dy}")
        self._display_type = StrParam('O', 'regular')
        input_file.close()
        psf_file.close()
        self.parameters = [self._input, self._output, self._sigma, self._psf,
                           self._psf_scale, self._mask, self._constrains, self._image_region,
                           self._convolution_size, self._zeropoint, self._pixel_scale, self._display_type, self._mode]

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
    def image_size(self):
        value = re.split(r'\s+', self._image_region.value)
        s1, s2 = int(value[1]), int(value[3])
        return s1, s2

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
        chi2 = -1
        self._components = []
        with open(file_name, 'r') as file:
            line = file.readline()
            while line:
                line = line.lstrip()
                if chi2 < 0 and line.startswith('#  Chi^2/nu = '):
                    chi2 = float(line.split('=')[1].split(',')[0])
                pos = line.find(')')
                if len(line) > 0 and pos > 0:
                    if line[0] == '0':
                        line = line.split(' ')
                        component = component_names[line[1].split('\n')[0]]()
                        file = component.read(file)
                        self._components.append(component)
                line = file.readline()
        return chi2

    def _galfit_output(self, str):
        state = 0
        for line in str.split('\n'):
            if state == 1:
                if line.startswith('======'):
                    state = 0
                else:
                    print(line)
            elif state == 2:
                if line.startswith('COUNTDOWN'):
                    state = 0
                else:
                    tmp += line + '\n'
            elif line.startswith('#  Input menu file:'):
                print(line)
            elif line.startswith('Initial parameters:'):
                print(line)
                state = 1
            elif line.startswith('Iteration :'):
                tmp = line + '\n'
                state = 2
        print(tmp)

    def run(self, galfit_file=None, galfit_mode=0):
        self.config.galfit_mode = galfit_mode
        if galfit_file is not None:
            with open(galfit_file, 'w') as file:
                print(self, file=file)
            result = subprocess.run(['./galfit', galfit_file],
                                    capture_output=True, text=True)
        else:
            result = subprocess.run(['./galfit'], input=str(self),
                                    capture_output=True, text=True)
        self._galfit_output(result.stdout)
        if result.stderr:
            print(result.stderr)
        result.check_returncode()

    def init_guess(self):
        with fits.open(self._config._input.value) as file:
            sky = Sky()
            # sky.background = _read_header(file[0], 'SKY')
            data = file[0].data
            # self.add_component(sky)
            sersic = Sersic()
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, (50, 50), filter_size=(
                3, 3), bkg_estimator=bkg_estimator)
            # print('Header background: ', sky.background,
            #       '\nEstimated background: ', bkg.background)
            sky.background = np.mean(bkg.background)
            print('Estimated background: ', sky.background)
            self.add_component(sky)
            threshold = 5 * bkg.background_rms

            ### 下面这段convolution是否必要？###
            from astropy.convolution import convolve
            from photutils.segmentation import make_2dgaussian_kernel
            kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0

            with fits.open(self.config._mask.value) as mask:
                mask_data = mask[0].data
            data = (data - bkg.background) * (1 - mask_data)

            convolved_data = convolve(data, kernel)
            ### ### ### ### ### ### ### ### ###

            # Using photutils.segmentation.SourceFinder to find initial stats
            finder = SourceFinder(npixels=10, progress_bar=False)
            segment_map = finder(convolved_data, threshold)
            # print(segment_map)
            if (segment_map == None):
                print("Source finding failure. Try reduce threshold.\n")
                return

            cat = SourceCatalog(data, segment_map,
                                convolved_data=convolved_data)
            print(cat)

            tbl = cat.to_table()
            tbl['xcentroid'].info.format = '.2f'  # optional format
            tbl['ycentroid'].info.format = '.2f'
            tbl['kron_flux'].info.format = '.2f'
            print(tbl)

            sersic.position = (round(_read_header(file[0], 'CEN_X', default = 200)),
                               round(_read_header(file[0], 'CEN_Y', default = 200)))
            # Seems like it is transposed
            map_label = segment_map.data[sersic.position[1],
                                         sersic.position[0]]
            
            if map_label == 0:
                #find the nearest component
                dists2 = []
                for i in range(len(tbl['xcentroid'])):
                    dists2.append((tbl['xcentroid'][i] - sersic.position[0])**2 + (tbl['ycentroid'][i] - sersic.position[1])**2)
                map_label = np.argmin(np.array(dists2)) + 1

            # Which one to use? kron or segment?
            total_flux = tbl['kron_flux'][map_label - 1]
            sersic.magnitude = -2.5 * \
                np.log10(total_flux) + self.config._zeropoint.value
            # ind = np.argmax(tbl['segment_flux'])

            sersic.effective_radius = round(
                np.sqrt(tbl['area'][map_label - 1].value))

            sersic.axis_ratio = 1-float(_read_header(file[0], 'ELL_E'))
            sersic.position_angle = float(_read_header(file[0], 'ELL_PA'))
            self.add_component(sersic)
