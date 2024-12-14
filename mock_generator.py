from components import Sky, Sersic
from task import GalfitTask, Config
from astropy.io import fits
import numpy as np
import numpy.random as random
import os


class ComponentGenerator:
    def __init__(self, cen_position, mag_limit, radius_limit):
        self.cen_position = cen_position
        self.mag_limit = mag_limit
        self.radius_limit = radius_limit
        self.pa_baseline = None
        self.count = None

    def __call__(self, count):
        self.pa_baseline = random.uniform(0, 180)
        self.count = count
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.count is None or self.count < 0:
            raise ValueError('count is not set properly')
        if self.count > 0:
            self.count -= 1
            return self._generate()
        else:
            self.pa_baseline = None
            self.count = None
            raise StopIteration

    def _generate(self):
        sersic = Sersic()
        sersic.magnitude = random.uniform(*self.mag_limit)
        sersic.effective_radius = random.uniform(*self.radius_limit)
        sersic.position = (self.cen_position[0] + random.uniform(-1, 1),
                           self.cen_position[1] + random.uniform(-1, 1))
        sersic.position_angle = (
            self.pa_baseline + random.uniform(-20, 20)) % 180
        if self.count == 2:
            sersic.set_sersic_index(0.5, False)  # bar
            sersic.axis_ratio = random.uniform(0.1, 0.5)
        elif self.count == 1:
            sersic.set_sersic_index(1, False)  # disk
            self.radius_limit = (self.radius_limit[0], sersic.effective_radius)
            sersic.axis_ratio = random.uniform(0.5, 1)
        elif self.count == 0:
            if random.uniform() < 0.5:
                sersic.set_sersic_index(4, False)
            else:
                sersic.set_sersic_index(random.uniform(2.5, 7), True)
            sersic.axis_ratio = random.uniform(0.5, 1)
        return sersic


class GalaxyGenerator:
    def __init__(self, out_dir, cen_pos, psf_file, mag_limit, radius_limit, sky_limit):
        self.out_dir = out_dir
        self.psf_file = psf_file
        self.sky_limit = sky_limit
        self.com_gen = ComponentGenerator(cen_pos, mag_limit, radius_limit)
        self.count = None

    def _generate(self):
        if not os.path.exists(f'{self.out_dir}/{self.count}/'):
            os.makedirs(f'{self.out_dir}/{self.count}/')
        config = Config(input_file=f'{self.out_dir}/blank.fits',
                        output_file=f'{self.out_dir}/{self.count}/input.fits',
                        psf_file=self.psf_file,
                        mask_file='none',
                        pixel_scale=0.396,
                        psf_scale=1,
                        zeropoint=24)
        task = GalfitTask(config)
        sky = Sky()
        sky.background = random.uniform(self.sky_limit)
        task.add_component(sky)
        count = random.randint(1, 4)
        if count == 3:
            print('with bar')
        # count = 3
        for comp in self.com_gen(count):
            task.add_component(comp)
        return task

    def __call__(self, count):
        self.count = count
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.count is None or self.count < 0:
            raise ValueError('count is not set properly')
        if self.count > 0:
            self.count -= 1
            return self._generate()
        else:
            self.count = None
            raise StopIteration


def noise_generator(data, sky_rms):
    shape = data.shape
    # calculate poisson noise for model
    poisson = np.random.poisson(data)
    # use sky rms to generate mock sky
    mock_sky = np.random.randn(shape[0], shape[1]) * sky_rms
    # final mock data = model with poisson noise + sky noise
    final = poisson + mock_sky
    return final


def gen_data(count):
    out_dir = './mock_galaxy'
    mag_limit = (12, 18)
    radius_limit = (10, 40)
    sky_limit = (0.8, 1.6)
    image_scale = (300, 300)
    psf_file = './S82/psf_r_cut65x65.fits'

    hdu = fits.PrimaryHDU(np.zeros(image_scale))
    hdu.writeto(f'{out_dir}/blank.fits', overwrite=True)
    galaxy_gen = GalaxyGenerator(out_dir, (image_scale[0]/2, image_scale[1]/2), psf_file,
                                 mag_limit, radius_limit, sky_limit)
    for task in galaxy_gen(count):
        task.run(galfit_mode=1)
        with fits.open(task.config.output_file) as hdul:
            data = hdul[0].data
        data = noise_generator(data, 0.01)
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(task.config.output_file, overwrite=True)


gen_data(5)


# def use_galfit_to_generate_mock_galaxy(input_par):
#     from astropy.nddata.utils import Cutout2D
#     band_list = ['g', 'r', 'i', 'z', 'y']

#     # cut a suitable size for fitting
#     # for j in range(5):
#     #     band = band_list[j]
#     #     with fits.open('PS1/{}_sky.fits'.format(band)) as hdu:
#     #         data = hdu[0].data
#     #         cut = Cutout2D(data, (50, 50), (75, 75))
#     #         cut.data -= np.mean(cut.data)
#     #         header = hdu[0].header
#     #         header['NCOMBINE'] = 1
#     #         hdu_temp = fits.PrimaryHDU(cut.data, header=header)
#     #         hdu_temp.writeto('PS1_mock/galfitm_configs/{}_bkg.fits'.format(band), overwrite=True)

#     # configs to generate mock galaxy model
#     Example = open(
#         'PS1_mock/galfitm_configs/Example_4_mock_generation.feedme', 'r')
#     content = Example.readlines()
#     Example.close()

#     for i in range(len(input_par)):
#         name = input_par[i]['Name']
#         content[42] = ' 9) {}      1          #  axis ratio (b/a)\n'.format(
#             input_par[i]['ratio'])
#         content[43] = '10) {}    1          #  position angle (PA) [deg: Up=0, Left=90]\n'.format(
#             input_par[i]['PA'])
#         for j in range(5):
#             band = band_list[j]
#             content[3] = 'A) PS1_mock/galfitm_configs/{}_bkg.fits            # Input data image (FITS file)\n'.format(
#                 band)
#             content[4] = 'B) PS1_mock/galfitm_results/pure_galaxy_model_5bands/{}_{}.fits       # Output data image block\n'.format(
#                 name, band)
#             content[6] = 'D) PS1/{}_psf.fits        # Input PSF image and (optional) diffusion kernel\n'.format(
#                 band)
#             content[36] = ' 3) {}     1          #  Integrated magnitude\n'.format(
#                 input_par[i]['Mag_{}'.format(band)])
#             content[37] = ' 4) {}      1          #  R_e (half-light radius)   [pix]\n'.format(
#                 input_par[i]['Re_{}'.format(band)])
#             content[38] = ' 5) {}      1          #  Sersic index n (de Vaucouleurs n=4)\n'.format(
#                 input_par[i]['n_{}'.format(band)])
#             temp = open(
#                 'PS1_mock/galfitm_configs/generate_mock_galaxy/{}_{}.feedme'.format(name, band), 'w')
#             temp.writelines(content)
#             temp.close()


# def prepare_mock_data(index_list, image_dir='galfitm/Gal/results/'):
#     # Use best-fit models as input, then
#     # Next, three noise processes are applied independently to the data: each pixel value p
#     # is replaced with a random number of events drawn from a Poisson distribution with mean
#     # p, Gaussian read-out noise, and a constant sky bkg with value equal to sigma of a circle
#     # with radius three times the re of AGN host (this encloses 96% and 79% of the total light for n=1 and n=4)

#     # table for sky rms
#     t = Table.read('catalog/AGN1_galaxy_EXPTIME.ipac', format='ascii.ipac')

#     if type(index_list) == int:
#         index_list = [index_list]

#     List = []
#     for Index in index_list:
#         # Index = 0
#         if os.path.exists('galfitm/Gal/mock/results/{}_mock_results.ipac'.format(Index)):
#             continue
#         print(Index)
#         List.append(Index)
#         # load best-fit results
#         with fits.open(image_dir+'{}_SS.fits'.format(Index), memmap=False) as hdu:
#             # total model = psf + sersic
#             model_g = hdu['COMPONENT_2_sersic_g'].data
#             model_r = hdu['COMPONENT_2_sersic_r'].data
#             model_i = hdu['COMPONENT_2_sersic_i'].data
#             model_z = hdu['COMPONENT_2_sersic_z'].data
#             model_y = hdu['COMPONENT_2_sersic_y'].data
#             # shape of the data
#             shape = model_g.shape
#             # get GAIN keyword
#             GAIN_g = hdu['INPUT_g'].header['GAIN']
#             GAIN_r = hdu['INPUT_r'].header['GAIN']
#             GAIN_i = hdu['INPUT_i'].header['GAIN']
#             GAIN_z = hdu['INPUT_z'].header['GAIN']
#             GAIN_y = hdu['INPUT_y'].header['GAIN']

#         # get sky rms
#         g_sky_rms = t[Index]['sky_rms_g']
#         r_sky_rms = t[Index]['sky_rms_r']
#         i_sky_rms = t[Index]['sky_rms_i']
#         z_sky_rms = t[Index]['sky_rms_z']
#         y_sky_rms = t[Index]['sky_rms_y']
#         # get sky var
#         g_sky_var = t[Index]['sky_var_sigma_g']
#         r_sky_var = t[Index]['sky_var_sigma_r']
#         i_sky_var = t[Index]['sky_var_sigma_i']
#         z_sky_var = t[Index]['sky_var_sigma_z']
#         y_sky_var = t[Index]['sky_var_sigma_y']

#         # load original config file
#         f = open('galfitm/Gal/configs/{}_SS.feedme'.format(Index), 'r')
#         content = f.readlines()
#         f.close()
#         low_bound_x = int(content[11].split(' ')[1])
#         low_bound_y = int(content[11].split(' ')[7])
#         x = float(content[43].split(' ')[2])
#         y = float(content[44].split(' ')[2])
#         while len(content) > 55:
#             del content[-1]

#         content[6] = 'C) none                # Sigma image name (made from data if blank or "none")\n'
#         content[9] = 'F) none                # Bad pixel mask (FITS image or ASCII coord list)\n'
#         content[11] = 'H) {} {} {} {}   # Image region to ' \
#                       'fit (xmin xmax ymin ymax)\n'.format(
#                           1, shape[1], 1, shape[0])
#         content[43] = ' 1) {}      1       # position x [pixel]\n'.format(
#             x-low_bound_x+1)
#         content[44] = ' 2) {}      1       # position y [pixel]\n'.format(
#             y-low_bound_y+1)

#         # for every source generate 50 mocks
#         for j in range(50):
#             np.random.seed(j)
#             # calculate poisson noise for model
#             poisson_g = apply_poisson_noise(np.abs(model_g * GAIN_g)) / GAIN_g
#             poisson_r = apply_poisson_noise(np.abs(model_r * GAIN_r)) / GAIN_r
#             poisson_i = apply_poisson_noise(np.abs(model_i * GAIN_i)) / GAIN_i
#             poisson_z = apply_poisson_noise(np.abs(model_z * GAIN_z)) / GAIN_z
#             poisson_y = apply_poisson_noise(np.abs(model_y * GAIN_y)) / GAIN_y
#             # use sky rms to generate mock sky
#             mock_sky_g = np.random.randn(shape[0], shape[1]) * g_sky_rms
#             mock_sky_r = np.random.randn(shape[0], shape[1]) * r_sky_rms
#             mock_sky_i = np.random.randn(shape[0], shape[1]) * i_sky_rms
#             mock_sky_z = np.random.randn(shape[0], shape[1]) * z_sky_rms
#             mock_sky_y = np.random.randn(shape[0], shape[1]) * y_sky_rms
#             # use sky var to generate bkg constant
#             sky_var_g = np.random.randn(1) * g_sky_var
#             sky_var_r = np.random.randn(1) * r_sky_var
#             sky_var_i = np.random.randn(1) * i_sky_var
#             sky_var_z = np.random.randn(1) * z_sky_var
#             sky_var_y = np.random.randn(1) * y_sky_var
#             # final mock data = model with poisson noise + sky noise
#             final_g = poisson_g + mock_sky_g
#             final_r = poisson_r + mock_sky_r
#             final_i = poisson_i + mock_sky_i
#             final_z = poisson_z + mock_sky_z
#             final_y = poisson_y + mock_sky_y
#             # save final mock data to fits file
#             hdu_temp_g = fits.PrimaryHDU(final_g, header=hdu['INPUT_g'].header)
#             hdu_temp_g.writeto(
#                 'galfitm/Gal/mock/data/{}_g_{}.fits'.format(Index, j), overwrite=True)
#             hdu_temp_r = fits.PrimaryHDU(final_r, header=hdu['INPUT_r'].header)
#             hdu_temp_r.writeto(
#                 'galfitm/Gal/mock/data/{}_r_{}.fits'.format(Index, j), overwrite=True)
#             hdu_temp_i = fits.PrimaryHDU(final_i, header=hdu['INPUT_i'].header)
#             hdu_temp_i.writeto(
#                 'galfitm/Gal/mock/data/{}_i_{}.fits'.format(Index, j), overwrite=True)
#             hdu_temp_z = fits.PrimaryHDU(final_z, header=hdu['INPUT_z'].header)
#             hdu_temp_z.writeto(
#                 'galfitm/Gal/mock/data/{}_z_{}.fits'.format(Index, j), overwrite=True)
#             hdu_temp_y = fits.PrimaryHDU(final_y, header=hdu['INPUT_y'].header)
#             hdu_temp_y.writeto(
#                 'galfitm/Gal/mock/data/{}_y_{}.fits'.format(Index, j), overwrite=True)

#             content[
#                 2] = 'A) galfitm/Gal/mock/data/{0}_g_{1}.fits,galfitm/Gal/mock/data/{0}_r_{1}.fits,galfitm/Gal/mock/data/{0}_i_{1}.fits,' \
#                      'galfitm/Gal/mock/data/{0}_z_{1}.fits,galfitm/Gal/mock/data/{0}_y_{1}.fits\n'.format(
#                          Index, j)
#             content[5] = 'B) galfitm/Gal/mock/results/{0}_{1}_SS.fits       # Output data image block\n'.format(
#                 Index, j)
#             content[36] = ' 1) {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}      0          #  sky background at center of fitting region [ADUs]\n'.\
#                 format(sky_var_g[0], sky_var_r[0],
#                        sky_var_i[0], sky_var_z[0], sky_var_y[0])

#             f = open(
#                 'galfitm/Gal/mock/configs/{}_{}_SS.feedme'.format(Index, j), 'w')
#             f.writelines(content)
#             f.close()

#         if len(List) == 5 or Index == len(index_list) - 1:
#             cores = multiprocessing.cpu_count()
#             pool = multiprocessing.Pool(processes=cores)
#             for each in List:
#                 for j in range(50):
#                     pool.apply_async(func=run_galfitm, args=(
#                         'galfitm/Gal/mock/configs/{}_{}_SS.feedme'.format(each, j),))
#             pool.close()
#             pool.join()
#             pool.terminate()

#             for each in List:
#                 obtain_mock_results(each)

#             List1 = glob('galfitm/Gal/mock/configs/*')
#             List2 = glob('galfitm/Gal/mock/data/*')
#             List3 = glob('galfitm/Gal/mock/results/*SS*')
#             os.remove('fit.log')
#             for each in List1:
#                 os.remove(each)
#             for each in List2:
#                 os.remove(each)
#             for each in List3:
#                 os.remove(each)
#             List = []
