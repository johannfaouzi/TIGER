"""Quant."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def _batch_exact_features_intervals(X, starts, ends, kind, offsets, q_positions, center_mask, out=None):
    """Exact feature computation, looping over the intervals.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_timepoints), dtype float64
            Batch of input time series.

    starts : ndarray of shape (n_intervals,), dtype int64
            Inclusive start index of each interval.

    ends : ndarray of shape (n_intervals,), dtype int64
        Exclusive end index of each interval.

    kind : ndarray of shape (n_intervals,), dtype int64
        Per-interval code:
            - 0 = raw passthrough (length-1 interval),
            - 1 = median only (no centering),
            - 2 = full quantile set (with centering).

    offsets : ndarray of shape (n_intervals + 1,), dtype int64
        CSR-style cumulative feature counts: interval `i` contributes `offsets[i+1] - offsets[i]` features, stored at
        `out[offsets[i]:offsets[i+1]]`.

    q_positions : ndarray of shape (n_features,), dtype float64
        Quantile position (in [0, 1]) requested for each output feature.

    center_mask : ndarray of shape (n_features,), dtype bool
        Whether each output feature must be centered (local interval mean subtracted) after being computed.

    out : ndarray of shape (n_samples, n_features), dtype float64, optional
        If provided, results are written directly into this array instead of allocating a new one.

    Returns
    -------
    out : ndarray of shape (n_samples, n_features), dtype float64
        Extracted features, one row per sample. The same object as the `out` argument if one was provided.
    """
    n_samples = X.shape[0]
    total = int(offsets[-1])
    if out is None:
        out = np.empty((n_samples, total), dtype=np.float64)
    n_intervals = len(starts)

    for i in range(n_intervals):
        s, e = int(starts[i]), int(ends[i])
        o0, o1 = int(offsets[i]), int(offsets[i + 1])

        if kind[i] == 0:
            # length-1 interval: raw value for every sample at once.
            out[:, o0] = X[:, s]
            continue

        m = e - s
        seg = X[:, s:e]
        seg_sorted = np.sort(seg, axis=-1)  # one vectorized sort for all samples

        # Every requested quantile position of this interval
        pos = q_positions[o0:o1] * (m - 1)  # shape (n_q,)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, m - 1)
        frac = pos - lo
        a = seg_sorted[:, lo]
        b = seg_sorted[:, hi]
        diff = b - a

        vals = np.where(frac >= 0.5, b - diff * (1.0 - frac), a + diff * frac)

        cmask = center_mask[o0:o1]
        if np.any(cmask):
            seg_mean = seg.mean(axis=-1, keepdims=True)
            vals[:, cmask] -= seg_mean

        out[:, o0:o1] = vals

    return out


class QuantTransformer(BaseEstimator, TransformerMixin):
    """Quant feature extraction algorithm.

    Parameters
    ----------
    depth : int, default=6
        The depth to stop extracting intervals at. Starting with the full series, the number of intervals extracted is
        ``2 ** depth`` (starting at 0) for each level.

    div : int, default=4
        The divisor to find the number of quantiles to extract from intervals.
    """

    def __init__(self, depth=6, div=4):
        self.depth = depth
        self.div = div

    @staticmethod
    def _repr_raw(X):
        """Raw representation."""
        return X

    @staticmethod
    def _repr_diff1_smooth(X):
        """Smoothed first-order difference representation."""
        d1 = np.diff(X, axis=-1)
        pad_width = [(0, 0)] * (d1.ndim - 1) + [(2, 2)]
        padded = np.pad(d1, pad_width, mode="edge")
        window = 5
        csum = np.cumsum(padded, axis=-1)
        csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,)), csum], axis=-1)
        smoothed = (csum[..., window:] - csum[..., :-window]) / window
        return np.ascontiguousarray(smoothed, dtype=np.float64)

    @staticmethod
    def _repr_diff2(X):
        """Second-order difference representation."""
        return np.ascontiguousarray(np.diff(X, n=2, axis=-1), dtype=np.float64)

    @staticmethod
    def _repr_fft_abs(X):
        """Magnitude of the real Fourier transform representation."""
        return np.ascontiguousarray(np.abs(np.fft.rfft(X, axis=-1)), dtype=np.float64)

    @classmethod
    def _representation_funcs(cls):
        """The 4 representation transform functions."""
        return (cls._repr_raw, cls._repr_diff1_smooth, cls._repr_diff2, cls._repr_fft_abs)

    def _representation_lengths(self):
        """Closed-form lengths for the 4 representations.

        Returns
        -------
        lengths : tuple of 4 int
            Lengths of the raw, smoothed first-difference, second-difference, and FFT-magnitude representations,
            respectively: (L, L - 1, L - 2, L // 2 + 1), where L is the length of each raw input time series.
        """
        return (self.n_timepoints_, self.n_timepoints_ - 1, self.n_timepoints_ - 2, self.n_timepoints_ // 2 + 1)

    def _make_intervals(self, length):
        """Method to generate the intervals.

        Returns
        -------
        starts, ends : ndarray of shape (n_intervals,), dtype int64

        depths : ndarray of shape (n_intervals,), dtype int64
            Depth level d (0 .. exponent - 1) each interval belongs to.
        """
        exponent = min(self.depth, int(np.log2(length)) + 1)

        starts_list, ends_list, depth_list = [], [], []
        for d in range(exponent):
            n = 2**d
            indices = np.linspace(0, length, n + 1).astype(np.int64)
            s, e = indices[:-1], indices[1:]
            starts_list.append(s)
            ends_list.append(e)
            depth_list.append(np.full(len(s), d, dtype=np.int64))

            if n > 1 and np.median(e - s) > 1:
                shift = int(np.ceil(length / n / 2))
                starts_list.append(s[:-1] + shift)
                ends_list.append(e[:-1] + shift)
                depth_list.append(np.full(len(s) - 1, d, dtype=np.int64))

        starts = np.concatenate(starts_list).astype(np.int64)
        ends = np.concatenate(ends_list).astype(np.int64)
        depths = np.concatenate(depth_list).astype(np.int64)
        return starts, ends, depths

    def _build_quantile_layout(self, starts, ends):
        """Determines the quantiles to compute in all the (sub-)intervals."""
        lengths = (ends - starts).astype(np.int64)
        n_intervals = len(starts)

        n_quantiles = np.zeros(n_intervals, dtype=np.int64)
        kind = np.zeros(n_intervals, dtype=np.int64)

        for i, n in enumerate(lengths):
            if n == 1:
                n_quantiles[i] = 1
                kind[i] = 0
            else:
                nq = 1 + (n - 1) // self.div
                n_quantiles[i] = nq
                kind[i] = 1 if nq == 1 else 2

        offsets = np.zeros(n_intervals + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(n_quantiles)
        total = int(offsets[-1])

        q_positions = np.full(total, 0.5, dtype=np.float64)
        center_mask = np.zeros(total, dtype=np.bool_)

        for i in range(n_intervals):
            o0, o1 = offsets[i], offsets[i + 1]
            if kind[i] == 2:
                q_positions[o0:o1] = np.linspace(0.0, 1.0, n_quantiles[i])
                center_mask[o0 + 1 : o1 : 2] = True
            elif kind[i] == 1:
                q_positions[o0:o1] = 0.5

        return n_quantiles, kind, offsets, center_mask, q_positions

    @staticmethod
    def _build_channel_permutation(offsets, n_channels):
        """Builds the index permutation that reorders a per-channel-then-per-feature layout.

        Parameters
        ----------
        offsets : ndarray of shape (n_intervals + 1,), dtype int64
            CSR-style cumulative single-channel feature counts (see `_build_quantile_layout`).

        n_channels : int
            Number of channels.

        Returns
        -------
        perm : ndarray of shape (n_channels * total,), dtype int64
            Channel permutation.
        """
        total = int(offsets[-1])
        n_intervals = len(offsets) - 1
        perm = np.empty(n_channels * total, dtype=np.int64)
        for i in range(n_intervals):
            o0, o1 = int(offsets[i]), int(offsets[i + 1])
            width = o1 - o0
            for c in range(n_channels):
                dest = o0 * n_channels + c * width
                source = c * total + o0
                perm[dest : dest + width] = np.arange(source, source + width)
        return perm

    def _compute_exact_features_one_representation(self, Z, layout, out):
        """Compute exact features for a single representation, writing them into `out`.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64
            Batch of input time series (`n_channels == 1` for a univariate input, normalized to 3D by `fit`/
            `transform`).

        layout : dict
            One entry of `self.layouts_` (keys "starts", "ends", "kind", "offsets", "center_mask", "q_positions",
            "channel_perm"), matching `Z`'s length.

        out : ndarray of shape (n_samples, n_channels * n_features_this_repr), dtype float64
            Destination slice (a view into the shared, pre-allocated output array built by
            `_compute_exact_features`) that this representation's features are written into. Channel-major
            then quantile-minor within each interval's block (see `_build_channel_permutation`).
        """
        n_samples, n_channels, length = Z.shape
        Z_flat = Z.reshape(n_samples * n_channels, length)

        if n_channels == 1:
            _batch_exact_features_intervals(
                Z_flat,
                layout["starts"],
                layout["ends"],
                layout["kind"],
                layout["offsets"],
                layout["q_positions"],
                layout["center_mask"],
                out,
            )
        else:
            block = _batch_exact_features_intervals(
                Z_flat,
                layout["starts"],
                layout["ends"],
                layout["kind"],
                layout["offsets"],
                layout["q_positions"],
                layout["center_mask"],
            )  # (n_samples * n_channels, total)

            total = block.shape[1]
            out[:, :] = block.reshape(n_samples, n_channels * total)[:, layout["channel_perm"]]
            del block

    def _compute_exact_features(self, X):
        """Compute exact quantiles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_timepoints), dtype float64, C-contiguous
            Batch of input time series.

        Returns
        -------
        out : ndarray of shape (n_samples, n_output_features_), dtype float64
            Exact quantiles.
        """
        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_output_features_), dtype=np.float64)
        offset = 0
        for repr_func, layout in zip(self._representation_funcs(), self.layouts_):
            width = int(layout["offsets"][-1]) * self.n_channels_
            self._compute_exact_features_one_representation(repr_func(X), layout, out[:, offset : offset + width])
            offset += width
        return out

    def _validate_input_shape(self, X, transform):
        """Validates the dimensions of X."""
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
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Batch of input time series, all of the same length.

        y : array-like of shape (n_samples,) or None, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : QuantTransformer
            The fitted transformer.
        """
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=False)
        self.n_channels_ = X.shape[1]
        self.n_timepoints_ = X.shape[2]

        self.layouts_ = []
        for length in self._representation_lengths():
            starts, ends, depths = self._make_intervals(length)
            _, kind, offsets, center_mask, q_positions = self._build_quantile_layout(starts, ends)
            channel_perm = self._build_channel_permutation(offsets, self.n_channels_) if self.n_channels_ > 1 else None
            self.layouts_.append(
                {
                    "starts": starts,
                    "ends": ends,
                    "depth": depths,
                    "kind": kind,
                    "offsets": offsets,
                    "center_mask": center_mask,
                    "q_positions": q_positions,
                    "channel_perm": channel_perm,
                }
            )

        self.n_output_features_ = sum(int(layout["offsets"][-1]) for layout in self.layouts_) * self.n_channels_

        return self

    def transform(self, X):
        """Applies feature extraction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Batch of input time series, of the same length and the same number of channels as seen in fit().

        Returns
        -------
        out : ndarray of shape (n_samples, n_output_features_), dtype float64
            Extracted QUANT features.
        """
        check_is_fitted(self, "layouts_")
        X = check_array(X, dtype=np.float64, order="C", ensure_2d=False, allow_nd=True)
        self._validate_input_shape(X, transform=True)
        return self._compute_exact_features(X)
