"""KG-MTP feature-extraction transform."""
from itertools import combinations
from typing import Tuple

import numpy as np
from numba import float32, njit, prange, vectorize
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Fixed kernel length used throughout the paper's kernel-grouping scheme:
# each of the 6 positions in a length-6 kernel is either positively
# weighted or not, and kernels are grouped by how many positions are
# positive (1 through 5 -- a kernel with 0 or 6 positive positions carries
# no discriminative information, so those two groups don't appear).
_KERNEL_LENGTH = 6

# Per-group multiplier applied to the weight matrix, for value_length 1
# through 5 (value_length 1-3 are left at x1). This is a fixed,
# paper-specific design choice reproduced verbatim from the weight
# construction that used to live inline in ``expri_code/run_ KGMTP.py``
# (``weights4 *= 2`` and ``weights5 *= 5``) -- it is not a general formula
# in ``_KERNEL_LENGTH``, so it's only meaningful for ``_KERNEL_LENGTH == 6``.
# Also drives the "alpha" multiplier in ``_build_kernel_group_tables``
# below (see that function for how the two connect).
_GROUP_MULTIPLIERS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 5.0}


def _build_kernel_group_tables():
    """Precompute the fixed per-kernel tables that ``_fit_biases`` and ``_transform`` both need.

    Returns
    -------
    indices : int32[:, :], shape (num_kernels, _KERNEL_LENGTH - 1)
        Row ``k`` holds kernel ``k``'s ``value_lengths[k]`` positive
        positions (as combinations of ``range(_KERNEL_LENGTH)``, the same
        combinatorial order ``_build_weights`` uses), left-padded with
        unused trailing zeros for kernels in smaller groups -- always safe
        to index with since callers only ever read the first
        ``value_lengths[k]`` columns of row ``k``.

    value_lengths, alpha_counts, gamma_counts : int32[:], shape (num_kernels,)
        Per-kernel ``value_length``, ``alpha``, and ``gamma`` (see above).

    group_sizes : int32[:], shape (_KERNEL_LENGTH - 1,)
        Number of kernels in each ``value_length`` group, in the same
        order the flat per-kernel tables are laid out in (all of group 1,
        then all of group 2, ...) -- this is also the exact layout
        ``_KGMTPBranch._build_weights`` concatenates its blocks in, so a
        kernel's row index into these tables is directly its row index
        into the fitted ``weights`` matrix.
    """
    kernel_length = _KERNEL_LENGTH
    max_value_length = kernel_length - 1

    rows, value_lengths, alpha_counts, gamma_counts, group_sizes = [], [], [], [], []
    for value_length in range(1, kernel_length):
        combos = list(combinations(range(kernel_length), value_length))
        group_sizes.append(len(combos))

        alpha = int(_GROUP_MULTIPLIERS[value_length])
        assert (alpha * kernel_length) % value_length == 0
        gamma = alpha * kernel_length // value_length

        for combo in combos:
            rows.append(list(combo) + [0] * (max_value_length - value_length))
            value_lengths.append(value_length)
            alpha_counts.append(alpha)
            gamma_counts.append(gamma)

    return (
        np.array(rows, dtype=np.int32),
        np.array(value_lengths, dtype=np.int32),
        np.array(alpha_counts, dtype=np.int32),
        np.array(gamma_counts, dtype=np.int32),
        np.array(group_sizes, dtype=np.int32),
    )


_INDICES, _VALUE_LENGTHS, _ALPHA_COUNTS, _GAMMA_COUNTS, _GROUP_SIZES = _build_kernel_group_tables()


# ---------------------------------------------------------------------------
# Numba kernels (module level -- see the module docstring for why)
# ---------------------------------------------------------------------------

@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    """Proportion-of-positive-values indicator: 1 if a > b else 0."""
    if a > b:
        return 1
    return 0


@njit(fastmath=True, cache=True, inline="always")
def _accumulate_alpha_gamma(A, G, C_alpha, C_gamma, dilation, padding, kernel_length):
    """Shift-and-accumulate ``A``/``G``."""
    start = dilation
    end = len(A) - padding
    for gamma_index in range(kernel_length // 2):
        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]
        end += dilation
    for gamma_index in range(kernel_length // 2 + 1, kernel_length):
        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]
        start += dilation


@njit(fastmath=True, parallel=False, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, weights, rng):
    """Fit one PPV bias threshold per (kernel, dilation, feature) triple."""
    num_examples = X.shape[0]
    kernel_length = _KERNEL_LENGTH
    num_kernels = weights.shape[0]
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        input_length = X.shape[1] + dilation

        A = np.zeros(input_length, dtype=np.float64)
        G = np.zeros(input_length, dtype=np.float64)
        padding = (kernel_length * dilation) // 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            value_length = _VALUE_LENGTHS[kernel_index]
            alpha = _ALPHA_COUNTS[kernel_index]
            gamma = _GAMMA_COUNTS[kernel_index]

            feature_index_end = feature_index_start + num_features_this_dilation
            _X1 = X[rng.integers(0, num_examples)]

            A[:-dilation] = -alpha * _X1
            G[:-dilation] = gamma * _X1

            C_alpha = np.zeros(input_length, dtype=np.float64)
            C_alpha[:] = A
            C_gamma = np.zeros((kernel_length, input_length), dtype=np.float64)
            C_gamma[kernel_length // 2] = G
            _accumulate_alpha_gamma(A, G, C_alpha, C_gamma, dilation, padding, kernel_length)

            i0, i1, i2, i3, i4 = (
                _INDICES[kernel_index, 0], _INDICES[kernel_index, 1], _INDICES[kernel_index, 2],
                _INDICES[kernel_index, 3], _INDICES[kernel_index, 4],
            )
            if value_length == 1:
                C1 = C_alpha + C_gamma[i0]
            elif value_length == 2:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1]
            elif value_length == 3:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
            elif value_length == 4:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3]
            else:
                C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3] + C_gamma[i4]
            C1 = C1.astype(np.float32)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C1, quantiles[feature_index_start:feature_index_end]
            )
            feature_index_start = feature_index_end

    return biases


@njit("(float64[:,:],Tuple((int32[:],int32[:],float32[:],float32[:,:])),int32)",
          fastmath=True, parallel=True, cache=True)
def _transform(X, parameters, n_features_per_kernel=5) -> Tuple[float32[:, :], float32[:, :]]:
    """Apply the fitted kernels, producing both PPV-pooling and Hydra features."""
    dilations, num_features_per_dilation, biases, weights = parameters
    kernel_length = weights.shape[1]

    num_examples = X.shape[0]
    num_kernels = len(weights)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    features = np.zeros((num_examples, num_features * n_features_per_kernel), dtype=np.float32)
    features_hydra = np.zeros((num_examples, num_kernels * num_dilations * 2), dtype=np.float32)

    for example_index in prange(num_examples):
        hydra_feature_index = 0
        feature_index_start = 0
        _X = X[example_index]

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            input_length = _X.shape[0] + dilation
            padding = (kernel_length * dilation) // 2
            output_length = _X.shape[0] + (2 * padding) - ((kernel_length - 1) * dilation)
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_hydra_max = np.zeros(num_kernels, dtype=np.float32)
            C_hydra_min = np.zeros(num_kernels, dtype=np.float32)

            kernel_offset = 0
            for group in range(kernel_length - 1):
                value_length = group + 1
                group_size = _GROUP_SIZES[group]
                alpha = _ALPHA_COUNTS[kernel_offset]
                gamma = _GAMMA_COUNTS[kernel_offset]

                A = np.zeros(input_length, dtype=np.float64)
                G = np.zeros(input_length, dtype=np.float64)
                A[:-dilation] = -alpha * _X
                G[:-dilation] = gamma * _X

                C_alpha = np.zeros(input_length, dtype=np.float64)
                C_alpha[:] = A
                C_gamma = np.zeros((kernel_length, input_length), dtype=np.float64)
                C_gamma[kernel_length // 2, :] = G
                _accumulate_alpha_gamma(A, G, C_alpha, C_gamma, dilation, padding, kernel_length)

                C_max = np.full(output_length, -np.inf)
                C_min = np.full(output_length, np.inf)
                C_max_index = np.zeros(output_length, dtype=np.int32)
                C_min_index = np.zeros(output_length, dtype=np.int32)

                for local_kernel_index in range(group_size):
                    kernel_index = kernel_offset + local_kernel_index
                    feature_index_end = feature_index_start + num_features_this_dilation

                    i0, i1, i2, i3, i4 = (
                        _INDICES[kernel_index, 0], _INDICES[kernel_index, 1], _INDICES[kernel_index, 2],
                        _INDICES[kernel_index, 3], _INDICES[kernel_index, 4],
                    )
                    if value_length == 1:
                        C1 = C_alpha + C_gamma[i0]
                    elif value_length == 2:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1]
                    elif value_length == 3:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
                    elif value_length == 4:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3]
                    else:
                        C1 = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] + C_gamma[i3] + C_gamma[i4]
                    C = C1.astype(np.float32)

                    for j in range(C.shape[0]):
                        if C_max[j] <= C[j]:
                            C_max[j] = C[j]
                            C_max_index[j] = kernel_index
                        if C_min[j] >= C[j]:
                            C_min[j] = C[j]
                            C_min_index[j] = kernel_index

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0
                        zero_count = 0
                        for j in range(C.shape[0]):
                            if (j < C.shape[0] - 1 and ((C[j] > _bias and C[j + 1] < _bias)
                                    or (C[j] < _bias and C[j + 1] > _bias))):
                                zero_count += 1

                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val

                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                        end = end + num_features
                        features[example_index, end] = zero_count / C.shape[0]
                    feature_index_start = feature_index_end

                for j in range(output_length):
                    C_hydra_max[C_max_index[j]] += C_max[j]
                    C_hydra_min[C_min_index[j]] += 1

                kernel_offset += group_size

            features_hydra[example_index, hydra_feature_index:hydra_feature_index + num_kernels] = C_hydra_max
            hydra_feature_index += num_kernels
            features_hydra[example_index, hydra_feature_index:hydra_feature_index + num_kernels] = C_hydra_min
            hydra_feature_index += num_kernels

    res = (features, features_hydra)
    return res


class _KGMTPBranch:
    """One kernel-grouping-and-pooling sub-transform."""

    def __init__(self, num_features=50_000, max_dilations_per_kernel=32, n_features_per_kernel=5):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

    def fit(self, X, rng):
        """Fit kernel dilations, quantile targets, and PPV bias thresholds."""
        weights = self._build_weights()
        num_kernels, kernel_length = weights.shape
        _, input_length = X.shape

        num_kernel_features = int(self.num_features / self.n_features_per_kernel)
        if num_kernel_features < num_kernels:
            raise ValueError(
                f"num_features // n_features_per_kernel must be at least "
                f"the number of kernels ({num_kernels}); got "
                f"{self.num_features} // {self.n_features_per_kernel} = "
                f"{num_kernel_features}. Increase `num_features`."
            )

        dilations, num_features_per_dilation = self._fit_dilations(
            input_length,
            num_kernel_features,
            self.max_dilations_per_kernel,
            num_kernels,
            kernel_length,
        )
        num_features_per_kernel = int(np.sum(num_features_per_dilation))
        quantiles = self._quantiles(num_kernels * num_features_per_kernel)

        biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles, weights, rng)

        self.dilations_ = dilations
        self.num_features_per_dilation_ = num_features_per_dilation
        self.biases_ = biases
        self.weights_ = weights
        self.n_features_in_ = input_length
        return self

    def transform(self, X):
        """Apply the fitted kernels to `X` (already a validated float64 2D array)."""
        parameters = (self.dilations_, self.num_features_per_dilation_, self.biases_, self.weights_)
        features, features_hydra = _transform(X, parameters, self.n_features_per_kernel)
        features = np.nan_to_num(features)
        return features, features_hydra

    def _build_weights(self):
        """Build the fixed kernel-grouping weight matrix."""
        length = _KERNEL_LENGTH
        blocks = []
        for value_length in range(1, length):
            combos = np.array(list(combinations(np.arange(length), value_length)), dtype=np.int32)
            num_kernels = len(combos)
            block = np.full((num_kernels, length), -1.0, dtype=np.float32)
            positive_value = (length - value_length) * 1 / value_length
            for i in range(num_kernels):
                for j in range(value_length):
                    block[i, combos[i, j]] = positive_value
            block *= _GROUP_MULTIPLIERS[value_length]
            blocks.append(block)
        return np.concatenate(blocks, axis=0)

    @staticmethod
    def _fit_dilations(input_length, num_features, max_dilations_per_kernel, num_kernels, kernel_length):
        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    @staticmethod
    def _quantiles(n):
        return np.array([(i * ((np.sqrt(5) + 1) / 2)) % 1 for i in range(1, n + 1)], dtype=np.float32)


class KGMTP(TransformerMixin, BaseEstimator):
    """KG-MTP transform: kernel-grouping features under multiple input transformations and pooling operators.

    Parameters
    ----------
    num_features : int, default=50_000
        Total PPV-pooling feature budget per channel, split evenly across the three
        branches (base, Hilbert, first-difference).

    max_dilations_per_kernel : int, default=32
        Maximum number of dilation per kernel.

    n_features_per_kernel : int, default=5
        Number of PPV-pooling statistics produced per kernel/dilation combination.

    random_state : None, int, or numpy.random.Generator, default=None
        Controls the random example draws used to fit each kernel's PPV bias threshold.
    """

    def __init__(self, num_features=50_000, max_dilations_per_kernel=32, n_features_per_kernel=5, random_state=None):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state

    def _validate_input_shape(self, X, transform):
        """Validate the dimensions of `X`."""
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_samples, n_channels, n_timepoints), got an array with {X.ndim} dimensions."
            )
        if transform:
            if X.shape[1] != self.n_channels_:
                raise ValueError(
                    f"Number of channels inconsistent with fit(): expected {self.n_channels_}, got {X.shape[1]}."
                )
            if X.shape[2] != self.n_timepoints_:
                raise ValueError(
                    f"Series length inconsistent with fit(): expected {self.n_timepoints_}, got {X.shape[2]}."
                )

    def fit(self, X, y=None):
        """Fit KG-MTP on `X`, discarding the resulting training features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input data.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit KG-MTP on `X` and return its transformed features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input data.

        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_ppv_features + n_hydra_features)
            Extracted features.
        """
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=False)
        n_channels = X.shape[1]
        self.n_channels_ = n_channels
        self.n_timepoints_ = X.shape[2]

        rng = self._check_random_state(self.random_state)
        num_features_per_branch = self.num_features // 3

        self.base_branches_ = []
        self.hilbert_branches_ = []
        self.diff_branches_ = []
        self.hydra_mu_ = []
        self.hydra_sigma_ = []
        self.n_ppv_features_per_channel_ = []
        self.n_hydra_features_per_channel_ = []

        channel_outputs = []
        for c in range(n_channels):
            X_c = np.ascontiguousarray(X[:, c, :])
            X_hilbert = self._hilbert_transform(X_c)
            X_diff = np.diff(X_c, 1)

            base = _KGMTPBranch(
                num_features_per_branch, self.max_dilations_per_kernel, self.n_features_per_kernel
            ).fit(X_c, rng)
            hilbert = _KGMTPBranch(
                num_features_per_branch, self.max_dilations_per_kernel, self.n_features_per_kernel
            ).fit(X_hilbert, rng)
            diff = _KGMTPBranch(
                num_features_per_branch, self.max_dilations_per_kernel, self.n_features_per_kernel
            ).fit(X_diff, rng)

            features, features_hydra_raw = self._transform_branches(base, hilbert, diff, X_c, X_hilbert, X_diff)
            mu, sigma = self._sparse_scaler_fit(features_hydra_raw)
            features_hydra = self._sparse_scaler_transform(features_hydra_raw, mu, sigma)

            self.base_branches_.append(base)
            self.hilbert_branches_.append(hilbert)
            self.diff_branches_.append(diff)
            self.hydra_mu_.append(mu)
            self.hydra_sigma_.append(sigma)
            self.n_ppv_features_per_channel_.append(features.shape[1])
            self.n_hydra_features_per_channel_.append(features_hydra.shape[1])

            channel_outputs.append(np.concatenate([features, features_hydra], axis=1))

        self.n_ppv_features_ = int(np.sum(self.n_ppv_features_per_channel_))
        self.n_hydra_features_ = int(np.sum(self.n_hydra_features_per_channel_))
        return np.concatenate(channel_outputs, axis=1)

    def transform(self, X):
        """Apply the already-fitted per-channel branches (and Hydra scalers) to `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input data. Must have the same number of channels and the same series length `X` was fitted on.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_ppv_features + n_hydra_features)
            Extracted features.
        """
        check_is_fitted(self, "base_branches_")
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=True)

        channel_outputs = []
        for c in range(self.n_channels_):
            X_c = np.ascontiguousarray(X[:, c, :])
            X_hilbert = self._hilbert_transform(X_c)
            X_diff = np.diff(X_c, 1)
            features, features_hydra_raw = self._transform_branches(
                self.base_branches_[c], self.hilbert_branches_[c], self.diff_branches_[c], X_c, X_hilbert, X_diff
            )
            features_hydra = self._sparse_scaler_transform(
                features_hydra_raw, self.hydra_mu_[c], self.hydra_sigma_[c]
            )
            channel_outputs.append(np.concatenate([features, features_hydra], axis=1))
        return np.concatenate(channel_outputs, axis=1)

    @staticmethod
    def _transform_branches(base, hilbert, diff, X, X_hilbert, X_diff):
        """Run one channel's three fitted branches and concatenate their raw output."""
        base_features, base_hydra = base.transform(X)
        hilbert_features, hilbert_hydra = hilbert.transform(X_hilbert)
        diff_features, diff_hydra = diff.transform(X_diff)

        features = np.concatenate([base_features, hilbert_features, diff_features], axis=1)
        features_hydra = np.concatenate([base_hydra, hilbert_hydra, diff_hydra], axis=1)
        return features, features_hydra

    @staticmethod
    def _sparse_scaler_fit(X, exponent=4):
        """Fit stats for the Hydra-feature scaler."""
        X = np.sqrt(np.clip(X, 0, None))
        epsilon = (X == 0).mean(axis=0) ** exponent + 1e-8
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) + epsilon
        return mu, sigma

    @staticmethod
    def _sparse_scaler_transform(X, mu, sigma, mask=True):
        """Apply the Hydra-feature scaler fit by ``_sparse_scaler_fit``."""
        X = np.sqrt(np.clip(X, 0, None))
        if mask:
            return ((X - mu) * (X != 0)) / sigma
        return (X - mu) / sigma

    def _check_random_state(self, random_state):
        """Coerce `random_state` into a `numpy.random.Generator`."""
        if random_state is None:
            return np.random.default_rng()
        if isinstance(random_state, (int, np.integer)):
            return np.random.default_rng(random_state)
        if isinstance(random_state, np.random.Generator):
            return random_state
        raise TypeError("random_state must be None, an int, or a numpy.random.Generator.")

    @staticmethod
    def _hilbert_transform(X):
        """Row-wise discrete Hilbert transform, vectorized across every row."""
        out = -signal.hilbert(X, axis=1).imag
        return out.astype(np.float32).astype(np.float64)
