from task import *
from plot_fig import *
import platform
import shutil

if __name__ == '__main__':
    name = 'NGC1326'

    os_name = platform.system()
    if os_name == 'Darwin':
        work_dir = './'
    elif os_name == 'Linux':
        work_dir = './'
    input = work_dir+'CGS/'+name+'/'+name+'_R_reg.fits'
    output = work_dir+'CGS/'+name+'/'+name+'_out.fits'
    psf = work_dir+'CGS/'+name+'/'+name+'_R_reg_ep.fits'
    mask = work_dir+'CGS/'+name+'/'+name+'_R_reg_mm.fits'
    comps = work_dir+'subcomps.fits'

    config = Config(input_file=input, output_file=output,
                    psf_file=psf, mask_file=mask)
    task = GalfitTask(config)
    task.read_component(work_dir+'galfit.08')

    # task.run()
    # task.run(log_file=None, galfit_mode=3)
    GalfitPlot(output, mask, comps, config.pixel_scale,
               config.zeropoint).plot(pro_1D=False)
    # GalfitPlot(output, comps).plot_comps()
