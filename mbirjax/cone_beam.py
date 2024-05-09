import warnings

import jax
import jax.numpy as jnp
from functools import partial
from mbirjax import TomographyModel
from jax import lax


class ConeBeamModel(TomographyModel) :
    """
    A class designed for handling forward and backward projections in a parallel beam geometry, extending the
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for parallel beam setups.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit parallel beam geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Args:
        angles (jnp.ndarray):
            A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different projection angles, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        source_detector_dist (float) – Distance between the X-ray source and the detector in units of ALU.
        det_channel_offset (float) – Distance = (projected center of rotation) - (center of detector channels) in ALU.
        det_row_offset (float) – Distance = (projected perpendicular to center of rotation) - (center of detector rows) in ALU.
        image_slice_offset (float) – Vertical offset of the image in ALU.
        angles (float, ndarray) – 1D array of view angles in radians.
        **kwargs (dict):
            Additional keyword arguments that are passed to the :ref:`TomographyModelDocs` constructor. These can
            include settings and configurations specific to the tomography model such as noise models or image dimensions.
            Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.
    """

    def __init__( self, sinogram_shape, source_detector_dist, det_row_offset = 0.0, recon_slice_offset = 0.0,
                  det_rotation = 0.0, angles, **kwargs ) :
        # Convert the view-dependent vectors to an array
        # This is more complicated than needed with only a single view-dependent vector but is included to
        # illustrate the process as shown in TemplateModel
        view_dependent_vecs = [vec.flatten() for vec in [angles]]
        try :
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e :
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")
        super().__init__(sinogram_shape, source_detector_dist=source_detector_dist, det_row_offset=det_row_offset,
                         recon_slice_offset=recon_slice_offset, view_params_array=view_params_array,
                         det_rotation=det_rotation, **kwargs)

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        if view_params_array.shape[0] != sinogram_shape[0] :
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

        recon_shape = self.get_params('recon_shape')
        if recon_shape[2] != sinogram_shape[1] :
            error_message = "Number of recon slices must match number of sinogram rows. \n"
            error_message += "Got {} for recon_shape and {} for sinogram_shape".format(recon_shape, sinogram_shape)
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for projection.

        Returns:
            List of required geometry parameters.
        """
        geometry_params = self.get_params(
            ['delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset', 'det_rotation',
             'source_detector_dist', 'magnification', 'delta_pixel_recon', 'recon_slice_offset'])

        return geometry_params

    @staticmethod
    def back_project_one_view_to_voxel(sinogram_view, pixel_index, angle, projector_params, coeff_power = 1):
        """
        Calculate the backprojection at a specified recon pixels given a sinogram view and model parameters.
        Also supports computation of the diagonal hessian when coeff_power = 2.

        Args:
            sinogram_view (jax array): one view of the sinogram to be back projected.
                2D jnp array of shape (num_det_rows)x(num_det_channels)
            pixel_index (jax array): set of pixels to which to back project.
                1D integer jnp array that indexes into flattened 2D recon.
                Should apply unravel_index(pixel_index, recon_shape) to recover i, j, k.
            angle (float): The angle in radians for this view.
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): Normally 1, but should be 2 when computing diagonal hessian 2.

        Returns:
            back_projection (jnp array): 1D array of length (number of pixels)*(number of slices)
        """
        # Get the part of the system matrix and channel indices for this voxel
        sinogram_view_shape = (1,) + sinogram_view.shape  # Adjoin a leading 1 to indicate a single view sinogram
        view_projector_params = (sinogram_view_shape,) + projector_params[1:]

        # Compute sparse system matrices for rows and columns
        # Bij_value, Bij_channel, Cij_value, Cij_row are all shaped [(num pixels)*(num slices)]x(2p+1)
        Bij_value, Bij_channel, Cij_value, Cij_row = ConeBeamModel.compute_sparse_Bij_Cij_single_view(pixel_index, angle, view_projector_params)

        # Determine shape of Bij where Nv = total number of voxels, and psf_width = 2p+1
        Nv, psf_width = Bij_value.shape  # This = (num pixels)*(num slices)

        # Generate full index arrays for rows and columns
        row_indices = Cij_row[:, :, None]  # Expand Cij_row to shape (num voxels, 2p+1, 1)
        col_indices = Bij_channel[:, None, :]  # Expand Bij_channel to shape (num voxels, 1, 2p+1)

        # Broadcast indices to shape (Nv, 2p+1, 2p+1)
        row_indices = jnp.broadcast_to(row_indices, shape=(Nv, psf_width, psf_width))
        col_indices = jnp.broadcast_to(col_indices, shape=(Nv, psf_width, psf_width))

        # Create sinogram_array with shape (Nv x psf_width x psf_width)
        sinogram_array = sinogram_view[row_indices, col_indices]

        # Broadcast Bij_value and Cij_row for element-wise multiplication
        Bij_value_expanded = Bij_value[:, :, None]  # Shape (Nv, 2p+1, 1)
        Cij_row_expanded = Cij_row[:, None, :]  # Shape (Nv, 1, 2p+1)

        # Compute back projection
        # coeff_power = 1 normally; coeff_power = 2 when computing diagonal of hessian
        back_projection = jnp.sum(sinogram_array * ((Bij_value_expanded * Cij_row_expanded) ** coeff_power), axis=(1, 2))

        return back_projection

    @staticmethod
    def forward_project_voxels_one_view(voxel_values, pixel_indices, angle, projector_params):
        """
        Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

        Args:
            voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            sinogram_shape (tuple): Sinogram shape (num_views, num_det_rows, num_det_channels)

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """
        if voxel_values.ndim != 2 :
            raise ValueError('voxel_values must have shape (num_indices, num_slices)')

        # Get the geometry parameters and the system matrix and channel indices
        num_views, num_det_rows, num_det_channels = projector_params[0]
        Bij_value, Bij_channel, Cij_value, Cij_row = ConeBeamModel.compute_sparse_Bij_Cij_single_view(pixel_indices, angle, projector_params)

        # Determine the size of the sinogram based on max indices + 1 for 0-based indexing
        Nr = num_det_rows
        Nc = num_det_channels

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((Nr, Nc))

        # Compute the outer products and scale by voxel_values
        # First, compute the outer product for each voxel's Bij_value
        outer_products = jnp.einsum('ki,kj->kij', Bij_value, Bij_value) * voxel_values[:, None, None]

        # Expand Cij_row and Cij_channel for broadcasting
        rows_expanded = Cij_row[:, :, None]  # Shape (Nv, 2p+1, 1)
        cols_expanded = Cij_row[:, None, :]  # Shape (Nv, 1, 2p+1)

        # Flatten the arrays to use in index_add
        flat_outer_products = outer_products.reshape(-1)
        flat_rows = rows_expanded.broadcast_to(outer_products.shape).reshape(-1)
        flat_cols = cols_expanded.broadcast_to(outer_products.shape).reshape(-1)

        # Aggregate the results into sinogram_view using index_add
        indices = (flat_rows, flat_cols)  # Prepare indices for index_add
        sinogram_view = lax.index_add(sinogram_view, indices, flat_outer_products)

        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnums=3)
    def compute_sparse_Bij_Cij_single_view( pixel_indices, angle, projector_params, p = 1 ):
        """
        Calculate the separable sparse system matrices for a subset of voxels and a single view.
        It returns a sparse matrix specified by the system matrix values and associated detector column index.
        Since this is for parallel beam geometry, the values are assumed to be the same for each row/slice.

        Args:
            pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
            angle (float):  Angle for this single view
            projector_params (tuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())
            p (int, optional, default=1):  # This is the assumed number of channels per side

        Returns:
            Bij_value, Bij_column, Cij_value, Cij_row (jnp array): Each with shape (num voxels)x(num slices)x(2p+1)
        """

        def recon_ijk_to_xyz( i, j, k, delta_pixel_recon, num_recon_rows, num_recon_cols, num_recon_slices,
                              recon_slice_offset, angle ) :
            # Compute the un-rotated coordinates relative to iso
            x_tilde = delta_pixel_recon * (i - (num_recon_rows - 1) / 2.0)
            y_tilde = delta_pixel_recon * (j - (num_recon_cols - 1) / 2.0)
            x = jnp.cos(angle) * x_tilde - jnp.sin(angle) * y_tilde  # corrected minus sign here
            y = jnp.sin(angle) * x_tilde + jnp.cos(angle) * y_tilde

            z = delta_pixel_recon * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset
            return x, y, z

        def geometry_xyz_to_uv_mag( x, y, z, source_detector_dist, magnification ) :
            # Compute the source to iso distance
            source_to_iso_dist = source_detector_dist / magnification

            # Check for a potential division by zero or very small denominator
            if (source_to_iso_dist - y) == 0 :
                raise ValueError("Invalid geometry: Denominator in pixel magnification calculation becomes zero.")

            # Compute the magnification at this specific voxel
            pixel_mag = source_detector_dist / (source_to_iso_dist - y)

            # Compute the physical position that this voxel projects onto the detector
            u = pixel_mag * x
            v = pixel_mag * z

            return u, v, pixel_mag

        def detector_uv_to_mn( u, v, det_rotation, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset,
                               num_det_rows, num_det_channels ) :
            # Account for small rotation of the detector
            u_tilde = jnp.cos(det_rotation) * u + jnp.sin(det_rotation) * v
            v_tilde = -jnp.sin(det_rotation) * u + jnp.cos(det_rotation) * v

            # Get the center of the detector grid for columns and rows
            det_center_channels = (num_det_channels - 1) / 2.0  # num_of_cols
            det_center_rows = (num_det_rows - 1) / 2.0  # num_of_rows

            # Calculate indices on the detector grid
            n = (u_tilde / delta_det_channel) + det_center_channels + det_channel_offset
            m = (v_tilde / delta_det_row) + det_center_rows + det_row_offset

            return m, n

        warnings.warn('Compiling for indices length = {}'.format(pixel_indices.shape))
        warnings.warn('Using hard-coded detectors per side.  These should be set dynamically based on the geometry.')

        # Get all the geometry parameters
        geometry_params = projector_params[2]
        delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, det_rotation, source_detector_dist, magnification, delta_pixel_recon, recon_slice_offset = geometry_params

        num_views, num_det_rows, num_det_channels = projector_params[0]
        num_recon_rows, num_recon_cols, num_recon_slices = projector_params[1][:]

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        recon_shape_2d = (num_recon_rows, num_recon_cols)
        num_of_indices = pixel_indices.size
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape_2d)

        # Replicate along the slice access
        i = (jnp.tile(row_index, reps=(num_recon_slices, 1)).T).flatten()
        j = (jnp.tile(col_index, reps=(num_recon_slices, 1)).T).flatten()
        k = (jnp.tile(jnp.arange(num_recon_slices), reps=(num_of_indices, 1))).flatten()

        # TODO: Need to check that i,j,k each have shape (num voxels)*(num slices)

        # All the following objects should have shape (num voxels)*(num slices)
        # x, y, z
        # u, v
        # mp, np
        # cone_angle_channel
        # cone_angle_row
        # cos_alpha_col
        # cos_alpha_row
        # W_col has shape
        # W_row has shape
        # mp has shape
        # np has shape 

        # Convert from ijk to coordinates about iso
        x, y, z = recon_ijk_to_xyz(i, j, k, delta_pixel_recon, num_recon_rows, num_recon_cols, num_recon_slices,
                                   recon_slice_offset, angle)

        # Convert from xyz to coordinates on detector
        u, v, pixel_mag = geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification)

        # Convert from uv to index coordinates in detector
        mp, np = detector_uv_to_mn(u, v, det_rotation, delta_det_channel, delta_det_row, det_channel_offset,
                                   det_row_offset, num_det_rows, num_det_channels)

        # Compute cone angle of pixel along columns and rows
        cone_angle_channel = jnp.arctan2(u, source_detector_dist)
        cone_angle_row = jnp.arctan2(v, source_detector_dist)

        # Compute cos alpha for row and columns
        cos_alpha_col = jnp.maximum(jnp.abs(jnp.cos(angle - cone_angle_channel)),
                                    jnp.abs(jnp.sin(angle - cone_angle_channel)))
        cos_alpha_row = jnp.maximum(jnp.abs(jnp.cos(cone_angle_row)), jnp.abs(jnp.sin(cone_angle_row)))

        # Compute projected voxel width along columns and rows
        W_col = pixel_mag * (delta_pixel_recon / delta_det_channel) * (cos_alpha_col / jnp.cos(cone_angle_channel))
        W_row = pixel_mag * (delta_pixel_recon / delta_det_row) * (cos_alpha_row / jnp.cos(cone_angle_row))

        # ################
        # Compute the Bij matrix entries
        # Compute a jnp channel index array with shape [(num voxels)*(num slices)]x1
        Bij_channel = jnp.round(mp).astype(int)
        Bij_channel = Bij_channel.reshape((-1, 1))

        # Compute a jnp channel index array with shape [(num voxels)*(num slices)]x(2p+1)
        Bij_channel = jnp.concatenate([Bij_channel + j for j in range(-p, p + 1)], axis=-1)

        # Compute the distance of each channel from the center of the voxel
        # Should be shape [(num voxels)*(num slices)]x(2p+1)
        delta_channel = jnp.abs(Bij_channel - mp.reshape((-1, 1)))

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        # Should be shape [(num voxels)*(num slices)]x(2p+1)
        tmp1 = (W_col + 1) / 2.0  # length = num_indices
        tmp2 = (W_col - 1) / 2.0  # length = num_indices
        L_channel = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta_channel), 0)

        # Compute Bij sparse matrix with shape [(num voxels)*(num slices)]x(2p+1)
        Bij_value = (delta_pixel_recon / cos_alpha_col) * (L_channel / delta_det_channel)
        Bij_value = Bij_value * (Bij_channel >= 0) * (Bij_channel < num_det_channels)

        # ################
        # Compute the Cij matrix entries
        # Compute a jnp row index array with shape [(num voxels)*(num slices)]x1
        Cij_row = jnp.round(mp).astype(int)
        Cij_row = Cij_row.reshape((-1, 1))

        # Compute a jnp row index array with shape [(num voxels)*(num slices)]x(2p+1)
        Cij_row = jnp.concatenate([Cij_row + j for j in range(-p, p + 1)], axis=-1)

        # Compute the distance of each row from the center of the voxel
        # Should be shape [(num voxels)*(num slices)]x(2p+1)
        delta_row = jnp.abs(Cij_row - mp.reshape((-1, 1)))

        # Calculate L = length of intersection between detector element and projection of flattened voxel
        # Should be shape [(num voxels)*(num slices)]x(2p+1)
        tmp1 = (W_row + 1) / 2.0  # length = num_indices
        tmp2 = (W_row - 1) / 2.0  # length = num_indices
        L_row = jnp.maximum(tmp1 - jnp.maximum(jnp.abs(tmp2), delta_row), 0)

        # Compute Cij sparse matrix with shape [(num voxels)*(num slices)]x(2p+1)
        Cij_value = (delta_pixel_recon / cos_alpha_col) * (L_row / delta_det_row)
        Cij_value = Cij_value * (Cij_row >= 0) * (Cij_row < num_det_rows)

        return Bij_value, Bij_channel, Cij_value, Cij_row