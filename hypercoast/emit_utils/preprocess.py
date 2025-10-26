"""Preprocessing and scaling utilities for hyperspectral data.

This module provides custom scalers for preprocessing hyperspectral remote sensing
data, including robust quantile-based scaling and logarithmic transformations.
"""

from typing import Tuple, Union

try:
    import torch
except ImportError:
    pass


class RobustMinMaxScaler:
    """Robust MinMax scaler using quantiles for outlier-resistant normalization.

    This scaler provides robust scaling by using quantiles instead of min/max values,
    making it less sensitive to outliers. It can operate in global or feature-wise mode.

    Args:
        feature_range: Target range (min, max) for scaled values.
        global_scale: If True, compute quantiles across all features globally.
            If False, compute quantiles independently for each feature.
        robust: If True, use quantiles for scaling. If False, use traditional min/max.
        quantile_range: Tuple of (lower, upper) quantiles (e.g., (0.25, 0.75) for IQR).
        clip_outliers: If True, clip values outside quantile range before scaling.

    Attributes:
        min_val: Fitted minimum (or lower quantile) value(s).
        max_val: Fitted maximum (or upper quantile) value(s).
        fitted: Whether the scaler has been fitted to data.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        global_scale: bool = True,
        robust: bool = True,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
        clip_outliers: bool = False,
    ):
        self.feature_range = feature_range
        self.global_scale = global_scale
        self.robust = robust
        self.quantile_range = quantile_range
        self.clip_outliers = clip_outliers
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(self, X: Union[torch.Tensor, "np.ndarray"]) -> "RobustMinMaxScaler":
        """Fit the scaler to the data by computing quantiles or min/max values.

        Args:
            X: Input tensor of shape (batch_size, features).

        Returns:
            self: Fitted scaler instance.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if self.robust:
            # Use quantiles for robust scaling
            if self.global_scale:
                # Global quantiles across all values
                flat_X = X.flatten()
                # If tensor is too large, use sampling for quantile calculation
                max_samples = 1000000  # 1M samples max
                if len(flat_X) > max_samples:
                    # Randomly sample from the tensor
                    indices = torch.randperm(len(flat_X))[:max_samples]
                    sampled_X = flat_X[indices]
                    self.min_val = torch.quantile(sampled_X, self.quantile_range[0])
                    self.max_val = torch.quantile(sampled_X, self.quantile_range[1])
                else:
                    self.min_val = torch.quantile(flat_X, self.quantile_range[0])
                    self.max_val = torch.quantile(flat_X, self.quantile_range[1])
            else:
                # Feature-wise quantiles
                self.min_val = torch.quantile(
                    X, self.quantile_range[0], dim=0, keepdim=True
                )
                self.max_val = torch.quantile(
                    X, self.quantile_range[1], dim=0, keepdim=True
                )
        else:
            # Use traditional min/max
            if self.global_scale:
                # Global min/max across all values
                self.min_val = torch.min(X)
                self.max_val = torch.max(X)
            else:
                # Feature-wise min/max
                self.min_val = torch.min(X, dim=0, keepdim=True)[0]
                self.max_val = torch.max(X, dim=0, keepdim=True)[0]

        self.fitted = True
        return self

    def transform(self, X: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        """Transform the data using fitted scaling parameters.

        Args:
            X: Input tensor of shape (batch_size, features).

        Returns:
            Scaled tensor in the target feature_range.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Clip outliers if using robust scaling
        if self.robust and self.clip_outliers:
            X = torch.clamp(X, min=self.min_val, max=self.max_val)

        # Avoid division by zero
        range_val = self.max_val - self.min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

        # Scale to [0, 1]
        scaled = (X - self.min_val) / range_val

        # Scale to desired range
        min_target, max_target = self.feature_range
        return scaled * (max_target - min_target) + min_target

    def fit_transform(self, X: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        """Fit the scaler and transform the data in one step.

        Args:
            X: Input tensor of shape (batch_size, features).

        Returns:
            Scaled tensor in the target feature_range.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        """Inverse transform scaled data back to original scale.

        Args:
            X: Scaled tensor.

        Returns:
            Tensor in original scale.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        min_target, max_target = self.feature_range

        # Scale back to [0, 1]
        normalized = (X - min_target) / (max_target - min_target)

        # Scale back to original range
        range_val = self.max_val - self.min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

        return normalized * range_val + self.min_val


class LogScaler:
    """Logarithmic scaler with optional shifting for non-positive values.

    This scaler applies log10 transformation after optionally shifting data to
    ensure all values are positive. Useful for compressing the dynamic range of
    water quality parameters that span multiple orders of magnitude.

    The transformation steps are:
    1. Optionally shift values so minimum becomes 0 (if shift_min=True).
    2. Add a small safety term to avoid log(0).
    3. Apply log10 transformation.

    Args:
        shift_min: Whether to shift data so the minimum value becomes 0.
        safety_term: Small constant added before log to avoid log(0).

    Attributes:
        global_min: Minimum value observed during fitting.
        shift_value: Amount to shift data (computed from global_min).
        fitted: Whether the scaler has been fitted to data.
    """

    def __init__(self, shift_min: bool = False, safety_term: float = 1e-8):
        self.safety_term = safety_term
        self.shift_min = shift_min  # Whether to shift minimum value to 0
        self.global_min = None
        self.shift_value = None
        self.fitted = False

    def fit(self, y: Union[torch.Tensor, "np.ndarray"]) -> "LogScaler":
        """Fit the scaler by computing the global minimum.

        Args:
            y: Input values (can be tensor or array).

        Returns:
            self: Fitted scaler instance.
        """
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        self.global_min = torch.min(y).item()
        # Calculate shift value to make minimum = 0
        self.shift_value = abs(self.global_min) if self.global_min < 0 else 0
        self.fitted = True
        return self

    def transform(self, y: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        """Transform data by applying log10 transformation.

        Args:
            y: Input values (can be tensor or array).

        Returns:
            Log-transformed values.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Step 1: Shift values so minimum becomes 0
        if self.shift_min:
            # Shift to make minimum = 0
            shifted = y - self.global_min
        else:
            # Use pre-calculated shift value
            shifted = torch.clamp(y, min=0)  # Ensure no negative values

        # Step 2: Add safety term to avoid log(0)
        safe_values = shifted + self.safety_term

        # Step 3: Apply log10
        log_values = torch.log10(safe_values)

        return log_values

    def fit_transform(self, y: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        """Fit the scaler and transform data in one step.

        Args:
            y: Input values (can be tensor or array).

        Returns:
            Log-transformed values.
        """
        return self.fit(y).transform(y)

    def inverse_transform(
        self, y_log: Union[torch.Tensor, "np.ndarray"]
    ) -> torch.Tensor:
        """Inverse transform log-transformed data back to original scale.

        Args:
            y_log: Log-transformed values.

        Returns:
            Values in original scale.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if not isinstance(y_log, torch.Tensor):
            y_log = torch.tensor(y_log, dtype=torch.float32)

        # Step 1: Apply 10^y
        exp_values = torch.pow(10, y_log)

        # Step 2: Remove safety term
        safe_removed = exp_values - self.safety_term

        # Step 3: Remove shift to restore original range
        if self.shift_min:
            original_values = safe_removed - self.global_min
        else:
            original_values = safe_removed

        return original_values
