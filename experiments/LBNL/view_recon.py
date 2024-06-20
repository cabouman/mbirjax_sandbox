import mbirjax
import numpy as np
import os

if __name__ == "__main__":

    plot_difference = False

    # Change to the directory containing the data.
    directory = '/scratch/gilbreth/buzzard/BLS-00637_dyparkinson/'

    filenames = ['recon.snr_db=30.sharpness=0.0.view_step=1.iters=20.npz', 'recon.snr_db=30.sharpness=2.0.view_step=1.iters=20.npz']
    full_paths = [os.path.join(directory, filename) for filename in filenames]

    recons = []
    for full_path in full_paths:
        with np.load(full_path) as data:
            recon = data['arr_0']
            recons.append(recon)

    if plot_difference:
        mbirjax.slice_viewer(recons[1] - recons[0])
    else:
        mbirjax.slice_viewer(recons[0], recons[1], title='Left:  ' + filenames[0] + '\nRight: ' + filenames[1])

