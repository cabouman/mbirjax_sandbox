import numpy as np
import time
from ruamel.yaml import YAML
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

import jax
import jax.numpy as jnp

import os
import dxchange
import h5py
import tomopy

# for install of h5py, dxchange and tomopy
# conda install h5py
# conda install -c conda-forge dxchange (or pip install dxchange)
# conda install -c conda-forge tomopy

if __name__ == "__main__":

    # Set the parameters for the experiment
    iterations_list = [10, 15, 20]
    snr_db = 30
    sharpness = 1.0
    view_step = 1
    recon_pad_factor = 1.15
    granularity = [1, 8, 64, 512, 2048]
    partition_sequence = [0, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 4]

    # Specify the input file:  <root_data_directory>/<input_data_directory>/<input_filename>
    input_filename = "nist-sand-30-200-mix_27keV_z8mm_n657_20240425_164409_.h5"
    input_sub_directory = "BLS-00637_dyparkinson"
    root_data_directory = "/depot/bouman/data/"
    input_directory = os.path.join(root_data_directory, input_sub_directory)

    # Specify the output directory: files will be in <wheretosave>/<input_data_directory>
    wheretosave = "/scratch/gilbreth/buzzard"
    output_directory = os.path.join(wheretosave, input_sub_directory)

    # Set the intensity window and define a subset of the sinogram
    vmin = 0
    vmax = 0.001
    num_slices_to_use = 5
    channels_cut = 2000  # Cut this many sinogram channels, half on each end.

    # Set up the output files for all iterations
    output_file_names = []
    param_file_names = []
    for num_iterations in iterations_list:
        file_name = f'recon.snr_db={snr_db}'
        file_name += f'.sharpness={sharpness}.view_step={view_step}.iters={num_iterations}'
        output_file_names.append(file_name + '.npz')
        param_file_names.append(file_name + '.params.yaml')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    full_input_path = os.path.join(input_directory, input_filename)
        
    detector_str = "/measurement/instrument/detector/"
    numslices = int(dxchange.read_hdf5(full_input_path, detector_str + "dimension_y")[0])
    numrays = int(dxchange.read_hdf5(full_input_path, detector_str + "dimension_x")[0])
    pxsize = dxchange.read_hdf5(full_input_path, detector_str + "pixel_size")[0] / 10.0  # /10 to convert units from mm to cm

    inst_str = "/measurement/instrument/"
    camera_str = inst_str + "camera_motor_stack/setup/"
    propagation_dist = dxchange.read_hdf5(full_input_path, camera_str + "camera_distance")[1]
    kev = dxchange.read_hdf5(full_input_path, inst_str + "monochromator/energy")[0] / 1000

    rotation_str = "/process/acquisition/rotation/"
    numangles = int(dxchange.read_hdf5(full_input_path, rotation_str + "num_angles")[0])
    angularrange = dxchange.read_hdf5(full_input_path, rotation_str + "range")[0]

    print(f'{input_filename}: \nslices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange}, \npxsize: {pxsize * 10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV')
    if kev > 100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')

    sinoused = (-1, num_slices_to_use, 1)  # using the whole numslices will make it run out of memory
    if sinoused[0] < 0:
        sinoused = (int(np.floor(numslices / 2.0) - np.ceil(sinoused[1] / 2.0)),
                    int(np.floor(numslices / 2.0) + np.floor(sinoused[1] / 2.0)), 1)

    tomo, flat, dark, anglelist = dxchange.exchange.read_aps_tomoscan_hdf5(full_input_path,
                                                                           sino=(sinoused[0], sinoused[1], sinoused[2]))
    anglelist = -anglelist

    tomo = tomo.astype(np.float32, copy=False)
    flat = flat.astype(np.float32, copy=False)
    dark = dark.astype(np.float32, copy=False)
    tomopy.normalize(tomo, flat, dark, out=tomo, ncore=64)
    tomopy.minus_log(tomo, out=tomo, ncore=64)

    start = channels_cut // 2
    stop = tomo.shape[2] + 1 - (channels_cut // 2)
    tomo = tomo[::view_step, :, start:stop]
    anglelist = anglelist[::view_step]

    cor = 1265.5
    theshift = int((cor - numrays / 2))
    sinogram = jax.device_put(tomo)

    # View sinogram
    # pu.slice_viewer(tomo.transpose((0, 2, 1)), title='Original sinogram')

    parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, anglelist)

    # Generate weights array
    weights = jax.numpy.ones_like(sinogram)  # parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, det_channel_offset=theshift, verbose=1, snr_db=snr_db)
    num_recon_rows = int(sinogram.shape[2] * recon_pad_factor)
    recon_shape = (num_recon_rows, num_recon_rows, sinogram.shape[1])
    parallel_model.set_params(recon_shape=recon_shape)

    parallel_model.set_params(granularity=granularity, partition_sequence=partition_sequence)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon = None
    init_recon = None
    first_iteration = 0
    for num_iterations, output_file_name, param_file_name in zip(iterations_list, output_file_names, param_file_names):
        recon, recon_params = parallel_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                   init_recon=init_recon, first_iteration=first_iteration)

        recon.block_until_ready()
        full_output_path = os.path.join(output_directory, output_file_name)
        np.savez_compressed(full_output_path, recon)

        recon_params = recon_params._asdict()
        mean_recon = jnp.mean(recon)
        mean_sino = jnp.mean(sinogram)
        recon_params['mean_reco'] = float(mean_recon)
        recon_params['mean_sino'] = float(mean_sino)
        recon_params['recon_shape'] = recon_shape
        recon_params['sino_shape'] = sinogram.shape

        full_output_path = os.path.join(output_directory, param_file_name)
        with open(full_output_path, 'w') as file:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(recon_params, file)

        # Return to original scaling for next round
        init_recon = recon
        first_iteration = num_iterations

    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Display results
    pu.slice_viewer(recon, title='VCD Recon ')

    a = 0

