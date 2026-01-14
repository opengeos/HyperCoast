"""Preprocessing utilities for MoE-VAE models.

This module provides custom scalers for robust data preprocessing,
including RobustMinMaxScaler and LogScaler for handling different
types of data transformations.
"""

try:
    import torch
except ImportError:
    torch = None


class RobustMinMaxScaler:
    """Robust MinMax scaler for PyTorch tensors with quantile-based scaling option.

    This scaler provides robust scaling by optionally using quantiles instead of
    min/max values, which reduces sensitivity to outliers. It supports both
    global and feature-wise scaling.

    Args:
        feature_range (tuple, optional): Desired range (min, max) for scaling.
            Defaults to (0, 1).
        global_scale (bool, optional): If True, scale using global quantiles
            across all features. If False, scale each feature independently.
            Defaults to True.
        robust (bool, optional): If True, use quantiles instead of min/max
            for robust scaling. Defaults to True.
        quantile_range (tuple, optional): Tuple of (lower, upper) quantiles
            to use (e.g., (0.05, 0.95)). Defaults to (0.25, 0.75).
        clip_outliers (bool, optional): If True, clip values outside quantile
            range to the quantile values. Defaults to False.
    """

    def __init__(
        self,
        feature_range=(0, 1),
        global_scale=True,
        robust=True,
        quantile_range=(0.25, 0.75),
        clip_outliers=False,
    ):
        """Initialize the RobustMinMaxScaler."""
        self.feature_range = feature_range
        self.global_scale = global_scale
        self.robust = robust
        self.quantile_range = quantile_range
        self.clip_outliers = clip_outliers
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(self, X):
        """Fit the scaler to the data.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features].

        Returns:
            RobustMinMaxScaler: The fitted scaler instance.
        """
        if torch is None:
            raise ImportError("PyTorch is required for RobustMinMaxScaler")

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

    def transform(self, X):
        """Transform the data using fitted parameters.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features].

        Returns:
            torch.Tensor: Scaled tensor.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        if torch is None:
            raise ImportError("PyTorch is required for RobustMinMaxScaler")

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

    def fit_transform(self, X):
        """Fit the scaler and transform the data in one step.

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, features].

        Returns:
            torch.Tensor: Scaled tensor.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transform the scaled data back to original scale.

        Args:
            X (torch.Tensor): Scaled tensor.

        Returns:
            torch.Tensor: Original scale tensor.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if torch is None:
            raise ImportError("PyTorch is required for RobustMinMaxScaler")

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
    """Special log scaler that handles negative values by shifting and log10.

    This scaler transforms data using log10 after optionally shifting values
    to ensure all values are positive. It's useful for data with wide dynamic
    ranges or skewed distributions.

    Steps:
        1. Optionally shift all values by adding abs(global_min) so minimum
           becomes 0
        2. Add a small safety term to avoid log(0)
        3. Apply log10 transformation

    Args:
        shift_min (bool, optional): Whether to shift minimum value to 0.
            Defaults to False.
        safety_term (float, optional): Small value added to avoid log(0).
            Defaults to 1e-8.
    """

    def __init__(self, shift_min=False, safety_term=1e-8):
        """Initialize the LogScaler."""
        self.safety_term = safety_term
        self.shift_min = shift_min
        self.global_min = None
        self.shift_value = None
        self.fitted = False

    def fit(self, y):
        """Fit the scaler to find the global minimum for shifting.

        Args:
            y (torch.Tensor or np.array): Input values.

        Returns:
            LogScaler: The fitted scaler instance.
        """
        if torch is None:
            raise ImportError("PyTorch is required for LogScaler")

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        self.global_min = torch.min(y).item()
        # Calculate shift value to make minimum = 0
        self.shift_value = abs(self.global_min) if self.global_min < 0 else 0
        self.fitted = True
        return self

    def transform(self, y):
        """Transform the data: shift to make min=0, add safety term, then log10.

        Args:
            y (torch.Tensor or np.array): Input values.

        Returns:
            torch.Tensor: Log-transformed values.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        if torch is None:
            raise ImportError("PyTorch is required for LogScaler")

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

    def fit_transform(self, y):
        """Fit and transform in one step.

        Args:
            y (torch.Tensor or np.array): Input values.

        Returns:
            torch.Tensor: Log-transformed values.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y_log):
        """Inverse transform: 10^y - safety_term - shift_value.

        Args:
            y_log (torch.Tensor or np.array): Log-transformed values.

        Returns:
            torch.Tensor: Original scale values.

        Raises:
            ValueError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if torch is None:
            raise ImportError("PyTorch is required for LogScaler")

        if not isinstance(y_log, torch.Tensor):
            y_log = torch.tensor(y_log, dtype=torch.float32)

        # Step 1: Apply 10^y
        exp_values = torch.pow(10, y_log)

        # Step 2: Remove safety term
        safe_removed = exp_values - self.safety_term

        # Step 3: Remove shift to restore original range
        if self.shift_min:
            original_values = safe_removed + self.global_min
        else:
            original_values = safe_removed

        return original_values
