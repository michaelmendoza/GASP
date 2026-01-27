"""
Multi-Shifted Dictionary approach for field map compensation in GASP.

This module implements Approach 0 from the field map compensation plan:
Generate signal models at multiple discrete field map offsets, then select/interpolate
the correct coefficients per-voxel based on the measured field map.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.interpolate import interp1d

from ..ssfp import ssfp
from ..gasp import train_gasp, _design_matrix, _to_matrix


class FieldMapDictionary:
    """
    Dictionary of GASP coefficients trained at discrete field map offsets.

    This class pre-computes GASP coefficients for a range of field map values,
    allowing per-voxel coefficient selection or interpolation at inference time.

    Parameters
    ----------
    field_offsets : npt.NDArray
        Array of field map offsets in Hz at which to compute coefficients.
    method : str
        GASP method: 'linear', 'affine', 'quad', or 'quad-cross'.
    useL2 : bool
        Whether to use L2 (ridge) regularization during training.
    lam : float
        L2 regularization parameter.

    Attributes
    ----------
    coefficients : dict[float, npt.NDArray]
        Dictionary mapping field offset (Hz) to coefficient array.
    field_offsets : npt.NDArray
        Sorted array of field map offsets.
    method : str
        GASP method used for training.
    _interpolator : interp1d or None
        Cached interpolator for coefficient lookup.
    """

    def __init__(
        self,
        field_offsets: npt.NDArray | None = None,
        method: str = "affine",
        useL2: bool = False,
        lam: float = 1e-2,
    ):
        if field_offsets is None:
            # Default: 21 bins from -200 to 200 Hz
            field_offsets = np.linspace(-200, 200, 21)

        self.field_offsets = np.sort(np.asarray(field_offsets))
        self.method = method
        self.useL2 = useL2
        self.lam = lam
        self.coefficients: dict[float, npt.NDArray] = {}
        self._interpolator: interp1d | None = None
        self._n_coeffs: int | None = None

    def train(
        self,
        T1: npt.NDArray,
        T2: npt.NDArray,
        TR: float,
        TE: float,
        alpha: npt.NDArray,
        pcs: npt.NDArray,
        D: npt.NDArray,
        f0: float = 0,
        M0: npt.NDArray | float = 1,
    ) -> "FieldMapDictionary":
        """
        Train GASP coefficients at each field map offset.

        Parameters
        ----------
        T1 : npt.NDArray
            Longitudinal relaxation times [H, W] in seconds.
        T2 : npt.NDArray
            Transverse relaxation times [H, W] in seconds.
        TR : float
            Repetition time in seconds.
        TE : float
            Echo time in seconds.
        alpha : npt.NDArray
            Flip angles [H, W] in radians.
        pcs : npt.NDArray
            Phase cycling values in radians.
        D : npt.NDArray
            Desired 1D spectral profile.
        f0 : float
            Base off-resonance frequency in Hz.
        M0 : npt.NDArray or float
            Proton density.

        Returns
        -------
        self : FieldMapDictionary
            Returns self for method chaining.
        """
        self.coefficients = {}

        for f_off in self.field_offsets:
            # Simulate SSFP signal at this field map offset
            M_sim = ssfp(
                T1, T2, TR, TE, alpha,
                dphi=pcs,
                field_map=f_off,
                M0=M0,
                f0=f0
            )

            # Train GASP coefficients
            _, A = train_gasp(
                M_sim, D,
                method=self.method,
                useL2=self.useL2,
                lam=self.lam
            )

            self.coefficients[float(f_off)] = A

        # Store number of coefficients for validation
        self._n_coeffs = len(next(iter(self.coefficients.values())))

        # Build interpolator for coefficient lookup
        self._build_interpolator()

        return self

    def train_from_signals(
        self,
        signals: dict[float, npt.NDArray],
        D: npt.NDArray,
    ) -> "FieldMapDictionary":
        """
        Train from pre-computed signals at different field map offsets.

        Parameters
        ----------
        signals : dict[float, npt.NDArray]
            Dictionary mapping field offset (Hz) to signal data [H, W, n_pcs].
        D : npt.NDArray
            Desired 1D spectral profile.

        Returns
        -------
        self : FieldMapDictionary
            Returns self for method chaining.
        """
        self.field_offsets = np.sort(np.array(list(signals.keys())))
        self.coefficients = {}

        for f_off in self.field_offsets:
            M_sim = signals[f_off]
            _, A = train_gasp(
                M_sim, D,
                method=self.method,
                useL2=self.useL2,
                lam=self.lam
            )
            self.coefficients[float(f_off)] = A

        self._n_coeffs = len(next(iter(self.coefficients.values())))
        self._build_interpolator()

        return self

    def _build_interpolator(self) -> None:
        """Build scipy interpolator for coefficient lookup."""
        if len(self.coefficients) < 2:
            self._interpolator = None
            return

        # Stack coefficients into array [n_offsets, n_coeffs]
        coeff_array = np.array([
            self.coefficients[f] for f in self.field_offsets
        ])

        # Create interpolator that handles complex coefficients
        # Interpolate real and imaginary parts separately
        self._interpolator = interp1d(
            self.field_offsets,
            coeff_array,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value=(coeff_array[0], coeff_array[-1])  # Clamp to edges
        )

    def get_coefficients(
        self,
        field_value: float,
        interpolation: Literal["nearest", "linear"] = "linear"
    ) -> npt.NDArray:
        """
        Get coefficients for a single field map value.

        Parameters
        ----------
        field_value : float
            Field map value in Hz.
        interpolation : str
            'nearest' for nearest-neighbor lookup, 'linear' for linear interpolation.

        Returns
        -------
        coefficients : npt.NDArray
            GASP coefficients for the given field map value.
        """
        if interpolation == "nearest":
            # Find nearest offset
            idx = np.argmin(np.abs(self.field_offsets - field_value))
            return self.coefficients[float(self.field_offsets[idx])]

        elif interpolation == "linear":
            if self._interpolator is None:
                raise ValueError("Interpolator not built. Call train() first.")
            return self._interpolator(field_value)

        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

    def apply(
        self,
        data: npt.NDArray,
        field_map: npt.NDArray,
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> npt.NDArray:
        """
        Apply field-map-compensated GASP to data.

        This method selects/interpolates coefficients per-voxel based on the
        local field map value and applies them to produce the output image.

        Parameters
        ----------
        data : npt.NDArray
            Complex SSFP data, shape [H, W, n_pcs].
        field_map : npt.NDArray
            Off-resonance frequency map in Hz, shape [H, W].
        interpolation : str
            'nearest' for nearest-neighbor, 'linear' for linear interpolation.

        Returns
        -------
        output : npt.NDArray
            Reconstructed image, shape [H, W].
        """
        if data.shape[:2] != field_map.shape:
            raise ValueError(
                f"Data shape {data.shape[:2]} does not match "
                f"field map shape {field_map.shape}"
            )

        H, W = data.shape[:2]
        output = np.zeros((H, W), dtype=data.dtype)

        # Convert data to matrix form and build design matrix
        X, _ = _to_matrix(data)
        Phi = _design_matrix(X, self.method)

        # Flatten field map
        field_flat = field_map.ravel()

        # Apply per-voxel
        for idx in range(H * W):
            f_local = field_flat[idx]
            A_local = self.get_coefficients(f_local, interpolation=interpolation)
            output.ravel()[idx] = Phi[idx] @ A_local

        return output

    def apply_vectorized(
        self,
        data: npt.NDArray,
        field_map: npt.NDArray,
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> npt.NDArray:
        """
        Vectorized application of field-map-compensated GASP.

        This is a faster implementation that uses vectorized operations
        instead of per-voxel loops. Uses binning for efficiency.

        Parameters
        ----------
        data : npt.NDArray
            Complex SSFP data, shape [H, W, n_pcs].
        field_map : npt.NDArray
            Off-resonance frequency map in Hz, shape [H, W].
        interpolation : str
            'nearest' or 'linear'.

        Returns
        -------
        output : npt.NDArray
            Reconstructed image, shape [H, W].
        """
        if data.shape[:2] != field_map.shape:
            raise ValueError(
                f"Data shape {data.shape[:2]} does not match "
                f"field map shape {field_map.shape}"
            )

        H, W = data.shape[:2]

        # Convert data to matrix form and build design matrix
        X, _ = _to_matrix(data)
        Phi = _design_matrix(X, self.method)
        field_flat = field_map.ravel()

        if interpolation == "nearest":
            return self._apply_nearest_vectorized(Phi, field_flat, H, W)
        else:
            return self._apply_linear_vectorized(Phi, field_flat, H, W)

    def _apply_nearest_vectorized(
        self,
        Phi: npt.NDArray,
        field_flat: npt.NDArray,
        H: int,
        W: int
    ) -> npt.NDArray:
        """Vectorized nearest-neighbor application."""
        output = np.zeros(H * W, dtype=Phi.dtype)

        # Assign each voxel to nearest bin
        bin_indices = np.argmin(
            np.abs(field_flat[:, None] - self.field_offsets[None, :]),
            axis=1
        )

        # Process each bin
        for bin_idx, f_off in enumerate(self.field_offsets):
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue

            A = self.coefficients[float(f_off)]
            output[mask] = Phi[mask] @ A

        return output.reshape(H, W)

    def _apply_linear_vectorized(
        self,
        Phi: npt.NDArray,
        field_flat: npt.NDArray,
        H: int,
        W: int
    ) -> npt.NDArray:
        """Vectorized linear interpolation application."""
        output = np.zeros(H * W, dtype=Phi.dtype)

        # Get interpolated coefficients for all voxels at once
        # Shape: [H*W, n_coeffs]
        A_all = self._interpolator(field_flat)

        # Element-wise multiply Phi with A and sum
        # Phi: [H*W, n_coeffs], A_all: [H*W, n_coeffs]
        output = np.sum(Phi * A_all, axis=1)

        return output.reshape(H, W)

    def save(self, filepath: str) -> None:
        """
        Save the dictionary to a numpy file.

        Parameters
        ----------
        filepath : str
            Path to save the dictionary (will be saved as .npz).
        """
        np.savez(
            filepath,
            field_offsets=self.field_offsets,
            coefficients=np.array([
                self.coefficients[f] for f in self.field_offsets
            ]),
            method=self.method,
            useL2=self.useL2,
            lam=self.lam
        )

    @classmethod
    def load(cls, filepath: str) -> "FieldMapDictionary":
        """
        Load a dictionary from a numpy file.

        Parameters
        ----------
        filepath : str
            Path to the saved dictionary (.npz file).

        Returns
        -------
        dictionary : FieldMapDictionary
            Loaded dictionary object.
        """
        data = np.load(filepath, allow_pickle=True)

        dict_obj = cls(
            field_offsets=data['field_offsets'],
            method=str(data['method']),
            useL2=bool(data['useL2']),
            lam=float(data['lam'])
        )

        # Populate coefficients
        coeff_array = data['coefficients']
        for i, f_off in enumerate(dict_obj.field_offsets):
            dict_obj.coefficients[float(f_off)] = coeff_array[i]

        dict_obj._n_coeffs = coeff_array.shape[1]
        dict_obj._build_interpolator()

        return dict_obj


def train_fieldmap_dictionary(
    T1: npt.NDArray,
    T2: npt.NDArray,
    TR: float,
    TE: float,
    alpha: npt.NDArray,
    pcs: npt.NDArray,
    D: npt.NDArray,
    field_range: tuple[float, float] = (-200, 200),
    n_bins: int = 21,
    method: str = "affine",
    useL2: bool = False,
    lam: float = 1e-2,
    f0: float = 0,
    M0: npt.NDArray | float = 1,
) -> FieldMapDictionary:
    """
    Convenience function to train a field map dictionary.

    Parameters
    ----------
    T1 : npt.NDArray
        Longitudinal relaxation times [H, W] in seconds.
    T2 : npt.NDArray
        Transverse relaxation times [H, W] in seconds.
    TR : float
        Repetition time in seconds.
    TE : float
        Echo time in seconds.
    alpha : npt.NDArray
        Flip angles [H, W] in radians.
    pcs : npt.NDArray
        Phase cycling values in radians.
    D : npt.NDArray
        Desired 1D spectral profile.
    field_range : tuple[float, float]
        Range of field map offsets in Hz (min, max).
    n_bins : int
        Number of discrete bins for the dictionary.
    method : str
        GASP method: 'linear', 'affine', 'quad', or 'quad-cross'.
    useL2 : bool
        Whether to use L2 (ridge) regularization.
    lam : float
        L2 regularization parameter.
    f0 : float
        Base off-resonance frequency in Hz.
    M0 : npt.NDArray or float
        Proton density.

    Returns
    -------
    dictionary : FieldMapDictionary
        Trained field map dictionary.
    """
    field_offsets = np.linspace(field_range[0], field_range[1], n_bins)

    dictionary = FieldMapDictionary(
        field_offsets=field_offsets,
        method=method,
        useL2=useL2,
        lam=lam
    )

    dictionary.train(
        T1=T1,
        T2=T2,
        TR=TR,
        TE=TE,
        alpha=alpha,
        pcs=pcs,
        D=D,
        f0=f0,
        M0=M0
    )

    return dictionary


def apply_gasp_with_fieldmap(
    data: npt.NDArray,
    field_map: npt.NDArray,
    dictionary: FieldMapDictionary,
    interpolation: Literal["nearest", "linear"] = "linear",
    vectorized: bool = True,
) -> npt.NDArray:
    """
    Apply GASP with field map compensation using a pre-trained dictionary.

    Parameters
    ----------
    data : npt.NDArray
        Complex SSFP data, shape [H, W, n_pcs].
    field_map : npt.NDArray
        Off-resonance frequency map in Hz, shape [H, W].
    dictionary : FieldMapDictionary
        Pre-trained field map dictionary.
    interpolation : str
        'nearest' for nearest-neighbor, 'linear' for linear interpolation.
    vectorized : bool
        If True, use vectorized (faster) implementation.

    Returns
    -------
    output : npt.NDArray
        Reconstructed image, shape [H, W].
    """
    if vectorized:
        return dictionary.apply_vectorized(data, field_map, interpolation)
    else:
        return dictionary.apply(data, field_map, interpolation)
