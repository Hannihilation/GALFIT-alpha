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
        sersic.position_angle = (self.pa_baseline + random.uniform(-20, 20)) % 180
        if self.count == 2:
            sersic.set_sersic_index(0.5, False) # bar
            sersic.axis_ratio = random.uniform(0.1, 0.5)
        elif self.count == 1:
            sersic.set_sersic_index(1, False) # disk
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
        # count = random.randint(1, 3)
        count = 3
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

gen_data(5)





def use_galfit_to_generate_mock_galaxy(input_par):
    from astropy.nddata.utils import Cutout2D
    band_list = ['g', 'r', 'i', 'z', 'y']

    # cut a suitable size for fitting
    # for j in range(5):
    #     band = band_list[j]
    #     with fits.open('PS1/{}_sky.fits'.format(band)) as hdu:
    #         data = hdu[0].data
    #         cut = Cutout2D(data, (50, 50), (75, 75))
    #         cut.data -= np.mean(cut.data)
    #         header = hdu[0].header
    #         header['NCOMBINE'] = 1
    #         hdu_temp = fits.PrimaryHDU(cut.data, header=header)
    #         hdu_temp.writeto('PS1_mock/galfitm_configs/{}_bkg.fits'.format(band), overwrite=True)

    # configs to generate mock galaxy model
    Example = open('PS1_mock/galfitm_configs/Example_4_mock_generation.feedme', 'r')
    content = Example.readlines()
    Example.close()

    for i in range(len(input_par)):
        name = input_par[i]['Name']
        content[42] = ' 9) {}      1          #  axis ratio (b/a)\n'.format(input_par[i]['ratio'])
        content[43] = '10) {}    1          #  position angle (PA) [deg: Up=0, Left=90]\n'.format(input_par[i]['PA'])
        for j in range(5):
            band = band_list[j]
            content[3] = 'A) PS1_mock/galfitm_configs/{}_bkg.fits            # Input data image (FITS file)\n'.format(band)
            content[4] = 'B) PS1_mock/galfitm_results/pure_galaxy_model_5bands/{}_{}.fits       # Output data image block\n'.format(name, band)
            content[6] = 'D) PS1/{}_psf.fits        # Input PSF image and (optional) diffusion kernel\n'.format(band)
            content[36] = ' 3) {}     1          #  Integrated magnitude\n'.format(input_par[i]['Mag_{}'.format(band)])
            content[37] = ' 4) {}      1          #  R_e (half-light radius)   [pix]\n'.format(input_par[i]['Re_{}'.format(band)])
            content[38] = ' 5) {}      1          #  Sersic index n (de Vaucouleurs n=4)\n'.format(input_par[i]['n_{}'.format(band)])
            temp = open('PS1_mock/galfitm_configs/generate_mock_galaxy/{}_{}.feedme'.format(name, band), 'w')
            temp.writelines(content)
            temp.close()