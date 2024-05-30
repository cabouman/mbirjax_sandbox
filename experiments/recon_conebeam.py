import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_views = 32
    num_det_rows = 32
    num_det_channels = 64
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    start_angle = -(np.pi + detector_cone_angle) * (1/2)
    end_angle = (np.pi + detector_cone_angle) * (1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up parallel beam model
    cone_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Here are other things you might want to do

    # Change the recon shape, which has the form (rows, columns, slices)
    # All else being equal, a smaller recon shape will result in a smaller projection on the detector.
    change_recon_shape = False
    if change_recon_shape:
        recon_shape = cone_model.get_params('recon_shape')
        recon_shape = (64, 64, 32)  # tuple(dim // 2 for dim in recon_shape)  # Or set to whatever you like
        cone_model.set_params(recon_shape=recon_shape)

    # Change the voxel side length (voxels are cubes)
    # The default detector side length is 1.0 (arbitrary units), and delta_voxel has the same units
    # If you change the voxel size, you might want to view the sinogram to see the results.
    # All else being equal, a smaller voxel size will result in a smaller projection on the detector.
    change_voxel_pitch = False
    if change_voxel_pitch:
        cone_model.set_params(delta_voxel=3.0)

    cone_model.set_params(det_channel_offset=10.5)    # You can change the center-of-rotation in the sinogram
    # cone_model.set_params(granularity=[1, 2, 8, 64, 256], partition_sequence=[0, 0, 1, 2, 3, 4, 2, 3, 2, 3, 3, 3, 3, 3, 3]) # You can change the partition sequence and granularity

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = cone_model.gen_modified_3d_sl_phantom()
    mbirjax.slice_viewer(phantom)
    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = cone_model.forward_project(phantom)
    # del phantom

    # View sinogram
    pu.slice_viewer(sinogram.transpose((1, 2, 0)), title='Original sinogram', slice_label='View')

    # Generate weights array
    weights = cone_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    cone_model.set_params(sharpness=sharpness, verbose=1)
    # cone_model.set_params(positivity_flag=True)

    # Print out model parameters
    cone_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    print('Starting recon')
    time0 = time.time()
    recon, fm_rmse = cone_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################
    with open('recon timing.txt', 'a') as f:
        print('\n-------------------------', file=f)
        print('Current stats:', file=f)
        print('Sinogram shape = {}'.format(sinogram.shape), file=f)
        print('Recon shape = {}'.format(recon.shape), file=f)
        print('Elapsed time for recon is {:.3f} seconds'.format(elapsed), file=f)
        mbirjax.get_memory_stats(print_results=True, file=f)
        print('-------------------------', file=f)

    # Display results
    # pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')
    pu.slice_viewer(recon, title='VCD Recon')

    # You can also display individual slides with the sinogram
    #pu.display_slices(phantom, sinogram, recon)
