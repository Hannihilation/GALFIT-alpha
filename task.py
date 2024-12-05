from components import *
from astropy.io import fits
import subprocess
import time
import os
from photutils.segmentation import SourceFinder, SourceCatalog
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.visualization as vis
import photutils.isophote as iso
from photutils.aperture import EllipticalAperture


def _read_header(hdu, key, default=None):
    if key in hdu.header:
        return hdu.header[key]
    elif '_'+key[1:] in hdu.header:
        return hdu.header['_'+key[1:]]
    if default is None:
        raise KeyError(f'Keyword {key} not found in header')
    return default


class Config:
    def __init__(self, input_file, output_file=None, psf_file=None, sigma_file='none',
                 mask_file=None, psf_scale=None, zeropoint=None, pixel_scale=None, galaxy_range = None):
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
        if psf_scale is None:
            psf_scale = _read_header(psf_file[0], 'SCALE')
        self._psf_scale = StrParam('E', psf_scale)
        self._constrains = StrParam('G', 'none')
        in_s1 = _read_header(input_file[0], 'NAXIS1')
        in_s2 = _read_header(input_file[0], 'NAXIS2')
        self._image_region = StrParam('H', f"1 {in_s1} 1 {in_s2}")
        self._galaxy_range = galaxy_range

        psf_s1 = _read_header(psf_file[0], 'NAXIS1')
        psf_s2 = _read_header(psf_file[0], 'NAXIS2')
        self._convolution_size = StrParam('I', f"{psf_s1} {psf_s2}")
        if zeropoint is None:
            zeropoint = _read_header(input_file[0], 'ZPT_GSC', default=24)
        self._zeropoint = StrParam('J', zeropoint)
        if pixel_scale is None:
            # print('pixel scale is None')
            cd11 = _read_header(input_file[0], 'CD1_1')
            cd12 = _read_header(input_file[0], 'CD1_2', default=0)
            cd21 = _read_header(input_file[0], 'CD2_1', default=0)
            cd22 = _read_header(input_file[0], 'CD2_2')
            dx = np.sqrt(cd11**2+cd12**2)
            dy = np.sqrt(cd21**2+cd22**2)
        else:
            # print('pixel scale found')
            dx, dy = pixel_scale, pixel_scale
        self._pixel_scale = StrParam('K', f"{dx} {dy}")
        #### DEBUG ####
        # print(self._pixel_scale)
        # print(self.pixel_scale)
        ###############
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
    def input_file(self):
        return self._input.value

    @property
    def output_file(self):
        return self._output.value

    @property
    def mask_file(self):
        return self._mask.value

    @property
    def psf_file(self):
        return self._psf.value

    @property
    def sigma_file(self):
        return self._sigma.value

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

    @property
    def galaxy_range(self):
        return self._galaxy_range

    def __repr__(self) -> str:
        s = ''
        for param in self.parameters:
            s += param.__repr__() + '\n'
        return s

    def _plot_model(self, hdu, ax, cut_coeff=None, min_max=None, is_origin=False, cut_margin = False):
        # if plot_type == 'data':
        #     input_file = fits.open(self.input_file)
        #     data = input_file[0].data
        # elif plot_type == 'model':
        #     output_file = fits.open(self.output_file)
        #     data = output_file[2].data

        data = hdu.data

        if is_origin:
            with fits.open(self.mask_file) as mask:
                mask_data = mask[0].data
                data = data * (1-mask_data)
        min_value = np.min(data)
        offset = abs(min_value) if min_value < 0 else 0
        data += offset
        if cut_coeff is not None:
            interval = vis.PercentileInterval(cut_coeff)
        elif min_max is not None:
            interval = vis.ManualInterval(*min_max)
        else:
            interval = vis.MinMaxInterval()
        norm = vis.ImageNormalize(data, interval=interval,
                                  stretch=vis.LogStretch(), clip=True)
        if cut_margin and self._galaxy_range is not None:
            l_margin, r_margin = self.galaxy_range
            data = data[l_margin:r_margin, l_margin:r_margin]
        # print(l_margin,r_margin)
        # print(len(data))
        ax.imshow(data, cmap='gray', origin='lower', norm=norm)

    def _plot_1Dpro(self, hdu, axs, types, label=None, is_origin=False,
                    is_comp=False, show_iso=False, sky = None, sma = 20, eps = 0.7, pa = 0, minsma = 5, maxsma = None, step=0.05, fix_center=False):
        data = hdu.data
        if is_origin:
            with fits.open(self.mask_file) as mask:
                data = np.ma.array(data, mask=(mask[0].data == 1))

        if (sky is not None) and (not is_comp):
            data -= sky

        mid_x = data.shape[0] / 2
        mid_y = data.shape[1] / 2

        x0, y0 = (_read_header(hdu, 'CEN_X', default = mid_x),
                               _read_header(hdu, 'CEN_Y', default = mid_y))

        # if self._cen_pos is None:
        #     x0 = data.shape[0] / 2
        #     y0 = data.shape[1] / 2
        # else:
        #     x0, y0 = self._cen_pos

        geometry = iso.EllipseGeometry(x0=x0, y0=y0, sma=sma,
                                       eps=eps, pa=pa)
        ellipse = iso.Ellipse(data, geometry=geometry)
        # ellipse = iso.Ellipse(data)
        isolist = ellipse.fit_image(
            minsma=minsma, maxsma=maxsma, step=step,
            fix_center=fix_center)
        sma_list = isolist.sma
        sma_list = sma_list * self.pixel_scale

        intens = isolist.intens
        intens_err = isolist.int_err
        mu = -2.5 * np.log10(intens) + self.zeropoint
        mu_err = 2.5 / np.log(10) * intens_err / intens
        pa = (isolist.pa * 180 / np.pi - 90) % 180

        out_list = {'pa': pa, 'pa_err': isolist.pa_err * 180 / np.pi,
                    'eps': isolist.eps, 'eps_err': isolist.ellip_err,
                    'mu': mu, 'mu_err': mu_err}
        print(out_list)
        time.sleep(3)
        if len(pa) == 0:
            print('No meaningful fit was possible. Continue to next figure.')
            return
        for ax, type in zip(axs, types):
            if is_origin:
                ax.errorbar(sma_list, out_list[type], out_list[type+'_err'], fmt='o',
                            markersize=2, markeredgewidth=0.5, capsize=3, label=label)
                if type == 'pa':
                    ax.set_ylim(0, 180)
                elif type == 'eps':
                    ax.set_ylim(0, 1)
                elif type == 'mu':
                    ymargin = np.ptp(out_list['mu']) * 0.1
                    ymin = np.min(out_list['mu']) - ymargin
                    ymax = np.max(out_list['mu']) + ymargin
                    ax.set_ylim(ymax, ymin)
                xmax = np.max(sma_list) * 1.1
                ax.set_xlim(0, xmax)

            else:
                ax.plot(sma_list, out_list[type],
                        label=label, linestyle='--', linewidth=0.5)

        if show_iso:
            fig = plt.figure()
            ax = fig.add_subplot()
            self._plot_model(hdu, ax, cut_coeff=99.5)
            for i in range(5, len(sma_list), 5):
                aper = EllipticalAperture((isolist.x0[i], isolist.y0[i]),
                                          isolist.sma[i], isolist.sma[i] *
                                          (1 - isolist.eps[i]),
                                          isolist.pa[i])
                aper.plot(ax)
            fig.savefig('iso.pdf', format='pdf')
            # fig.show()

    def plot(self, cut_coeff=99.5, pro_1D=True, components=None):
        fig = plt.figure(figsize=(7, 7))
        gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)
        axs = np.array([[fig.add_subplot(gs[i, j])
                       for j in range(2)] for i in range(3)])
        if self._galaxy_range is None:
            # print('before: ', self.galaxy_range)
            GalfitTask(self).init_guess()
            # print('after: ', self.galaxy_range)
            # l_margin, r_margin = self.galaxy_range
        with fits.open(self.output_file) as model:
            for hdu in model[1:]:
                type = hdu.header['OBJECT']
                type.strip()
                if type == 'model':
                    print(f'Working on {type}')
                    self._plot_model(hdu, axs[1, 1], cut_coeff=cut_coeff, cut_margin=True)
                    if pro_1D:
                        self._plot_1Dpro(
                            hdu, axs[:, 0], ['eps', 'pa', 'mu'], label='model')
                elif type == 'residual map':
                    print(f'Working on {type}')
                    self._plot_model(hdu, axs[2, 1], cut_coeff=cut_coeff, is_origin=True, cut_margin=True)
                else:
                    print(f'Working on original data')
                    self._plot_model(
                        hdu, axs[0, 1], cut_coeff=cut_coeff, is_origin=True, cut_margin=True)
                    if pro_1D:
                        self._plot_1Dpro(
                            hdu, axs[:, 0], ['eps', 'pa', 'mu'], label='origin', is_origin=True, show_iso=True)

        if components is not None and pro_1D:
            with fits.open(components) as comps:
                for i, hdu in enumerate(comps[1:]):
                    type = hdu.header['OBJECT']
                    type.strip()
                    if type == 'sky':
                        continue
                    if type in component_names:
                        self._plot_1Dpro(
                            hdu, axs[2:, 0], ['mu'], label=type+str(i),
                            is_comp=True)

        axs[2, 0].legend()
        axs[0, 0].set_ylabel('$\epsilon$')
        axs[1, 0].set_ylabel('PA (degree)')
        axs[2, 0].set_ylabel('$\mu_R$ (mag/arcsec^2)')
        for a in axs[:, 1]:
            a.set_xticks([])
            a.set_yticks([])
        axs[0, 0].set_xticks([])
        axs[1, 0].set_xticks([])
        axs[2, 0].set_xlabel('Radius (arcsec)')

        # plt.show()
        fig_file = self.output_file.replace('.fits', '.pdf')
        fig.savefig(fig_file, format='pdf')

    def plot_comps(self, cut_coeff=99.5):
        with fits.open('./subcomps.fits') as comps:
            length = len(comps)
            fig, ax = plt.subplots(1, length)
            for i, hdu in enumerate(comps):
                self._plot_model(hdu, ax[i], cut_coeff=cut_coeff)
            plt.legend()
            # plt.show()
            fig_file = self.config.output_file.replace('.fits', '_comps.pdf')
            plt.savefig(fig_file, format='pdf')


class GalfitTask:
    FIGSIZE_TO_DIAMETER_LIMIT = 5

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

    def _galfit_output(self, str: str):
        state = 0
        returncode = 0
        tmp = ''
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
            elif line.startswith('   Doh!  GALFIT crashed'):
                returncode = 1
        print(tmp)
        return returncode

    def run(self, galfit_file=None, galfit_mode=0):
        self.config.galfit_mode = galfit_mode
        if galfit_file is None:
            galfit_file = self.config._output.value.replace('.fits', '.galfit')
        with open(galfit_file, 'w') as file:
            print(self, file=file)
        result = subprocess.run(['./galfit', galfit_file],
                                capture_output=True, text=True)
        crash = self._galfit_output(result.stdout)
        if result.stderr:
            print(result.stderr)
        # result.check_returncode()
        return crash

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

            if self.config._mask.value == 'none':
                mask_data = np.zeros_like(data)
            else:
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
                raise ValueError("Source finding failure. Try reduce threshold.")

            cat = SourceCatalog(data, segment_map,
                                convolved_data=convolved_data)
            print(cat)

            tbl = cat.to_table()
            tbl['xcentroid'].info.format = '.2f'  # optional format
            tbl['ycentroid'].info.format = '.2f'
            tbl['kron_flux'].info.format = '.2f'
            print(tbl['xcentroid', 'ycentroid', 'kron_flux', 'area'])

            mid_x = _read_header(file[0], 'NAXIS2')/2
            mid_y = _read_header(file[0], 'NAXIS1')/2

            position = (_read_header(file[0], 'CEN_X', default = mid_x),
                               _read_header(file[0], 'CEN_Y', default = mid_y))
            # Seems like it is transposed
            map_label = segment_map.data[round(position[1]),
                                         round(position[0])]
            if map_label == 0:
                #find the nearest component
                min_dist = 10000000
                for i in range(len(tbl['xcentroid'])):
                    dist = (tbl['xcentroid'][i] - position[0])**2 + (tbl['ycentroid'][i] - position[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        map_label = i + 1
            sersic.position = (tbl['xcentroid'][map_label - 1],
                              tbl['ycentroid'][map_label - 1])

            # Which one to use? kron or segment?
            total_flux = tbl['kron_flux'][map_label - 1]
            sersic.magnitude = -2.5 * \
                np.log10(total_flux) + self.config._zeropoint.value
            # ind = np.argmax(tbl['segment_flux'])

            sersic.effective_radius = round(
                np.sqrt(tbl['area'][map_label - 1].value))

            # Cut the graph to proper size
            # print('Mid: ', sersic.position[0], ' Radius: ', sersic.effective_radius, ' Fig_size: ', 2*mid_x )
            if sersic.position[0] + self.FIGSIZE_TO_DIAMETER_LIMIT * sersic.effective_radius < 2 * mid_x:
                self.config._galaxy_range = (round(sersic.position[1] - self.FIGSIZE_TO_DIAMETER_LIMIT * sersic.effective_radius), round(sersic.position[0] + self.FIGSIZE_TO_DIAMETER_LIMIT * sersic.effective_radius))
            else:
                self.config._galaxy_range = (0, _read_header(file[0], 'NAXIS1'))

            sersic.axis_ratio = 1-float(_read_header(file[0], 'ELL_E', default=0.25))
            sersic.position_angle = float(_read_header(file[0], 'ELL_PA', default=0))
            self.add_component(sersic)
