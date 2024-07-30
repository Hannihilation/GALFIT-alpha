import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
import astropy.visualization as vis
import photutils.isophote as iso
from photutils.aperture import EllipticalAperture
from components import *


class GalfitPlot:
    def __init__(self, model, mask, components=None, pixel_scale=1, zeropoint=0,
                 center_position=None, title=None, sma_init=100, eps_init=0.5,
                 pa_init=0, minsma=5, maxsma=None, step=0.05, fix_center=False):
        self._model = model
        self._mask = mask
        self._components = components
        self._title = title
        self._cen_pos = center_position
        self._pixel_scale = pixel_scale
        self._zeropoint = zeropoint
        self._sma = sma_init
        self._eps = eps_init
        self._pa = pa_init
        self._minsma = minsma
        self._maxsma = maxsma
        self._step = step
        self._fix_center = fix_center

        if self._components is not None:
            for hdu in fits.open(self._components):
                if hdu.header['OBJECT'] == 'sky':
                    self._sky = hdu.data
                    break
        else:
            self._sky = None

    def _plot_model(self, hdu, ax, cut_coeff=None, min_max=None, is_origin=False):
        data = hdu.data
        if is_origin:
            with fits.open(self._mask) as mask:
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
        ax.imshow(data, cmap='gray', origin='lower', norm=norm)

    def _plot_1Dpro(self, hdu, axs, types, label=None, is_origin=False,
                    is_comp=False, show_iso=False):
        data = hdu.data
        if is_origin:
            with fits.open(self._mask) as mask:
                data = np.ma.array(data, mask=(mask[0].data == 1))

        if (self._sky is not None) and (not is_comp):
            data -= self._sky

        if self._cen_pos is None:
            x0 = data.shape[0] / 2
            y0 = data.shape[1] / 2
        else:
            x0, y0 = self._cen_pos

        geometry = iso.EllipseGeometry(x0=x0, y0=y0, sma=self._sma,
                                       eps=self._eps, pa=self._pa)
        ellipse = iso.Ellipse(data, geometry=geometry)
        # ellipse = iso.Ellipse(data)
        isolist = ellipse.fit_image(
            minsma=self._minsma, maxsma=self._maxsma, step=self._step,
            fix_center=self._fix_center)
        sma_list = isolist.sma
        sma_list = sma_list * self._pixel_scale

        intens = isolist.intens
        intens_err = isolist.int_err
        mu = -2.5 * np.log10(intens) + self._zeropoint
        mu_err = 2.5 / np.log(10) * intens_err / intens
        pa = (isolist.pa * 180 / np.pi - 90) % 180

        out_list = {'pa': pa, 'pa_err': isolist.pa_err * 180 / np.pi,
                    'eps': isolist.eps, 'eps_err': isolist.ellip_err,
                    'mu': mu, 'mu_err': mu_err}
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
                    xmax = np.max(sma_list) * 1.1
                    ax.set_xlim(0, xmax)
                    ax.set_ylim(ymax, ymin)
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

    def plot(self, cut_coeff=99.5, pro_1D=True):
        fig = plt.figure(figsize=(7, 7))
        gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0)
        axs = np.array([[fig.add_subplot(gs[i, j])
                       for j in range(2)] for i in range(3)])
        with fits.open(self._model) as model:
            for hdu in model[1:]:
                type = hdu.header['OBJECT']

                type.strip()
                if type == 'model':
                    print(f'Working on {type}')
                    self._plot_model(hdu, axs[1, 1], cut_coeff=cut_coeff)
                    if pro_1D:
                        self._plot_1Dpro(
                            hdu, axs[:, 0], ['eps', 'pa', 'mu'], label='model')
                elif type == 'residual map':
                    print(f'Working on {type}')
                    self._plot_model(hdu, axs[2, 1], cut_coeff=cut_coeff)
                else:
                    print(f'Working on original data')
                    self._plot_model(
                        hdu, axs[0, 1], cut_coeff=cut_coeff, is_origin=True)
                    if pro_1D:
                        self._plot_1Dpro(
                            hdu, axs[:, 0], ['eps', 'pa', 'mu'], label='origin', is_origin=True, show_iso=True)

        if self._components is not None and pro_1D:
            with fits.open(self._components) as comps:
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
        fig_file = self._model.replace('.fits', '.pdf')
        fig.savefig(fig_file, format='pdf')

    # def plot(self, cut_coeff=99.5, pro_1D=True):
    #     fig, ax = plt.subplots(3, 2)
    #     with fits.open(self.__model__) as model:
    #         for hdu in model[1:]:
    #             type = hdu.header['OBJECT']
    #             type.strip()
    #             if type == 'model':
    #                 self.__plot_model__(hdu, ax[1, 1], cut_coeff=cut_coeff)
    #                 if pro_1D:
    #                     self.__plot_1Dpro__(
    #                         hdu, ax[:, 0], ['pa', 'eps', 'mu'], label='model')
    #             elif type == 'residual map':
    #                 self.__plot_model__(hdu, ax[2, 1], cut_coeff=cut_coeff)
    #             else:
    #                 self.__plot_model__(hdu, ax[0, 1], cut_coeff=cut_coeff)
    #                 if pro_1D:
    #                     self.__plot_1Dpro__(
    #                         hdu, ax[:, 0], ['pa', 'eps', 'mu'], label='origin')
    #     if self.__components__ is not None and pro_1D:
    #         with fits.open(self.__components__) as comps:
    #             for i, hdu in enumerate(comps[1:]):
    #                 type = hdu.header['OBJECT']
    #                 type.strip()
    #                 if type == 'sky':
    #                     continue
    #                 if type in component_names:
    #                     self.__plot_1Dpro__(
    #                         hdu, ax[2:, 1], ['mu'], label=type+str(i))

    #     plt.legend()
    #     ax[0, 0].set_ylabel('PA')
    #     ax[1, 0].set_ylabel('EPS')
    #     ax[2, 0].set_ylabel('Intensity')
    #     for a in ax[:, 1]:
    #         a.set_xticks([])
    #         a.set_yticks([])
    #     ax[0, 0].set_xticks([])
    #     ax[1, 0].set_xticks([])
    #     fig.subplots_adjust(wspace=0, hspace=0)

    #     # plt.show()
    #     fig_file = self.__model__.replace('.fits', '.pdf')
    #     plt.savefig(fig_file, format='pdf')

    def plot_comps(self, cut_coeff=99.5):
        if self._components is None:
            return
        with fits.open(self._components) as comps:
            length = len(comps)
            fig, ax = plt.subplots(1, length)
            for i, hdu in enumerate(comps):
                self._plot_model(hdu, ax[i], cut_coeff=cut_coeff)
            plt.legend()
            # plt.show()
            fig_file = self._model.replace('.fits', '_comps.pdf')
            plt.savefig(fig_file, format='pdf')
