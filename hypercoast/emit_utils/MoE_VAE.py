"""Mixture of Experts Variational Autoencoder (MoE-VAE) models.

This module implements VAE and MoE-VAE architectures for water quality parameter
estimation from hyperspectral remote sensing data. It includes sparse dispatching
for efficient expert routing and training/evaluation utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_lightning import LightningModule
    from torch.distributions.normal import Normal
    from torch.utils.data import DataLoader
except ImportError:
    pass


class VAE(LightningModule):
    """Variational Autoencoder for water quality parameter estimation.

    This class implements a standard VAE architecture with configurable encoder
    and decoder networks for estimating water quality parameters from hyperspectral
    remote sensing reflectance data.

    Args:
        input_dim: Number of input features (spectral bands).
        output_dim: Number of output features (water quality parameters).
        latent_dim: Dimension of the latent space.
        encoder_hidden_dims: List of hidden layer dimensions for the encoder.
        decoder_hidden_dims: List of hidden layer dimensions for the decoder.
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'leakyrelu').
        use_norm: Type of normalization ('batch', 'layer', 'group', or False).
        use_dropout: Whether to use dropout regularization.
        use_softplus_output: Whether to apply softplus activation to output.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        activation: str = "leakyrelu",
        use_norm: Union[str, bool] = False,
        use_dropout: bool = False,
        use_softplus_output: bool = False,
        **kwargs,
    ):
        super().__init__()
        # Define the activation function
        self.use_softplus_output = use_softplus_output
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder layers
        self.encoder_layers = self.build_layers(
            input_dim, encoder_hidden_dims, use_norm, use_dropout
        )
        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Decoder layers
        self.decoder_layers = self.build_layers(
            latent_dim, decoder_hidden_dims, use_norm, use_dropout
        )
        # self.decoder_layers.add_module('softplus', nn.Softplus())
        self.decoder_layers.add_module(
            "output_layer", nn.Linear(decoder_hidden_dims[-1], output_dim)
        )
        if self.use_softplus_output:
            self.decoder_layers.add_module("output_activation", nn.Softplus())
        # self.decoder_layers.add_module('output_activation', nn.Tanh())  # Assuming output is in range [-1, 1]
        # with the classic robust preprocessing method it is -1 to 1, but for others it may not.

    def build_layers(
        self,
        input_dim: int,
        hidden_dims: List[int],
        use_norm: Union[str, bool],
        use_dropout: bool = False,
    ) -> nn.Sequential:
        """Build a sequential network with specified layers and normalization.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            use_norm: Type of normalization ('batch', 'layer', 'group', or False).
            use_dropout: Whether to add dropout layers.

        Returns:
            Sequential module containing the network layers.
        """
        layers = []
        current_size = input_dim
        for hidden_dim in hidden_dims:
            next_size = hidden_dim
            layers.append(nn.Linear(current_size, next_size))
            if use_norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_norm == "group":
                num_groups = max(1, hidden_dim // 4)
                layers.append(
                    nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
                )
            layers.append(self.activation)
            if use_dropout:
                layers.append(nn.Dropout(0.1))
            current_size = next_size
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            mu: Mean of latent distribution.
            log_var: Log variance of latent distribution.
        """
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution.
            log_var: Log variance of latent distribution.

        Returns:
            z: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to output.

        Args:
            z: Latent vector.

        Returns:
            Decoded output tensor.
        """
        return self.decoder_layers(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Dictionary containing 'pred_y', 'mu', and 'log_var'.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred_y = self.decode(z)
        return {"pred_y": pred_y, "mu": mu, "log_var": log_var}

    def loss_fn(
        self, output_dict: Dict[str, torch.Tensor], kld_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss including reconstruction and KL divergence terms.

        Args:
            output_dict: Dictionary containing 'pred_y', 'y', 'mu', and 'log_var'.
            kld_weight: Weight for the KL divergence term.

        Returns:
            Dictionary containing 'total_loss', 'mae_loss', 'mse_loss', and 'kld_loss'.
        """
        pred_y, y, mu, log_var = (
            output_dict["pred_y"],
            output_dict["y"],
            output_dict["mu"],
            output_dict["log_var"],
        )
        batch_size = y.shape[0]
        MAE = F.l1_loss(pred_y, y, reduction="mean")
        # Reconstruction loss (MSE)
        MSE = F.mse_loss(pred_y, y, reduction="mean")
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        # Return combined loss
        return {
            "total_loss": MAE + kld_weight * KLD,
            "mae_loss": MAE,
            "mse_loss": MSE,
            "kld_loss": KLD,
        }


class SparseDispatcher(object):
    """Helper for implementing sparse mixture of experts routing.

    This class efficiently dispatches inputs to different experts based on
    gating weights and combines their outputs. It leverages sparsity by only
    sending batch elements to experts with non-zero gate values.

    The dispatcher performs two key operations:
    1. dispatch(): Distribute input samples to appropriate experts.
    2. combine(): Aggregate expert outputs weighted by gate values.

    Example:
        gates = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]])  # (batch=2, experts=3)
        dispatcher = SparseDispatcher(num_experts=3, gates=gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [expert(expert_inputs[i]) for i, expert in enumerate(experts)]
        combined_output = dispatcher.combine(expert_outputs)

    Args:
        num_experts: Total number of experts.
        gates: Tensor of shape (batch_size, num_experts) with gating weights.
            Non-zero values indicate which experts receive which samples.
    """

    def __init__(self, num_experts: int, gates: torch.Tensor):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts

        # Safety check: ensure at least one example per expert
        if (gates.sum(dim=0) == 0).any():
            # Find experts with no assignments and create dummy assignments
            empty_experts = (gates.sum(dim=0) == 0).nonzero().squeeze(1)
            if empty_experts.numel() > 0:
                # Assign the first example to all empty experts with a small weight
                for expert_idx in empty_experts:
                    gates[0, expert_idx] = 1e-5

        # Sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # Drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # Get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # Calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()

        # Safety check: ensure no expert has 0 examples
        for i, size in enumerate(self._part_sizes):
            if size == 0:
                # Add a dummy example to this expert
                self._part_sizes[i] = 1
                if i >= len(self._expert_index):
                    # Add a new dummy index if needed
                    self._expert_index = torch.cat(
                        [self._expert_index, torch.tensor([[i]], device=gates.device)]
                    )
                    self._batch_index = torch.cat(
                        [self._batch_index, torch.tensor([0], device=gates.device)]
                    )

        # Expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

        # Safety check for nonzero gates
        if (self._nonzero_gates <= 0).any():
            self._nonzero_gates = torch.clamp(self._nonzero_gates, min=1e-5)

    def dispatch(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """Dispatch input samples to their assigned experts.

        Args:
            inp: Input tensor of shape (batch_size, input_dim).

        Returns:
            List of tensors, one for each expert, containing the inputs
            assigned to that expert.
        """
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(
        self, expert_out: List[torch.Tensor], multiply_by_gates: bool = True
    ) -> torch.Tensor:
        """Combine expert outputs weighted by gate values.

        Args:
            expert_out: List of expert output tensors, each with shape
                (expert_batch_size_i, output_dim).
            multiply_by_gates: Whether to weight outputs by gate values.

        Returns:
            Combined output tensor of shape (batch_size, output_dim).
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self) -> List[torch.Tensor]:
        """Get gate values for each expert's assigned samples.

        Returns:
            List of tensors containing gate values for each expert's samples.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_VAE(LightningModule):
    """Mixture of Experts Variational Autoencoder for water quality estimation.

    This class implements a sparsely-gated MoE architecture where multiple VAE
    experts specialize in different regions of the input space. A learned gating
    network dynamically routes each input to the top-k most relevant experts.

    Args:
        input_dim: Number of input features (spectral bands).
        output_dim: Number of output features (water quality parameters).
        latent_dim: Dimension of the latent space for each expert VAE.
        encoder_hidden_dims: List of hidden layer dimensions for encoders.
        decoder_hidden_dims: List of hidden layer dimensions for decoders.
        num_experts: Total number of expert VAEs.
        k: Number of experts to activate for each input sample.
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'leakyrelu').
        noisy_gating: Whether to add noise to gating for exploration during training.
        use_norm: Type of normalization ('batch', 'layer', 'group', or False).
        use_dropout: Whether to use dropout regularization.
        use_softplus_output: Whether to apply softplus activation to output.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        num_experts: int,
        k: int = 4,
        activation: str = "leakyrelu",
        noisy_gating: bool = True,
        use_norm: Union[str, bool] = False,
        use_dropout: bool = False,
        use_softplus_output: bool = False,
        **kwargs,
    ):
        super(MoE_VAE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.num_experts = num_experts
        self.k = k
        self.activation = activation
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_softplus_output = use_softplus_output

        # instantiate experts
        self.experts = nn.ModuleList(
            [
                VAE(
                    self.input_dim,
                    self.output_dim,
                    self.latent_dim,
                    self.encoder_hidden_dims,
                    self.decoder_hidden_dims,
                    self.activation,
                    use_norm=self.use_norm,
                    use_dropout=self.use_dropout,
                    use_softplus_output=self.use_softplus_output,
                )
                for i in range(self.num_experts)
            ]
        )

        self.w_gate = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_dim, num_experts, dtype=self.dtype), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.batch_gates = None

        assert self.k <= self.num_experts

    def forward(
        self, x: torch.Tensor, moe_weight: float = 1e-2
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the MoE-VAE model.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            moe_weight: Weight for the MoE load balancing loss.

        Returns:
            Dictionary containing:
                - 'pred_y': Predicted output tensor (batch_size, output_dim).
                - 'moe_loss': Load balancing loss to encourage uniform expert usage.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        self.batch_gates = gates
        # calculate importance loss
        importance = gates.sum(0)

        moe_loss = moe_weight * self.cv_squared(
            importance
        ) + moe_weight * self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = []
        for i in range(self.num_experts):
            input_i = expert_inputs[i]
            if input_i.shape[0] > 1:
                expert_outputs.append(self.experts[i](input_i)["pred_y"])
            else:
                expert_outputs.append(
                    torch.zeros(
                        (input_i.shape[0], self.output_dim), device=input_i.device
                    )
                )
        pred_y = dispatcher.combine(expert_outputs)
        return {"pred_y": pred_y, "moe_loss": moe_loss}

    def loss_fn(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute MoE-VAE loss including reconstruction and load balancing terms.

        Args:
            output_dict: Dictionary containing 'pred_y', 'y', and 'moe_loss'.

        Returns:
            Dictionary containing 'total_loss', 'mae_loss', 'mse_loss', and 'moe_loss'.
        """
        pred_y = output_dict["pred_y"]
        y = output_dict["y"]
        batch_size = y.shape[0]
        MAE = F.l1_loss(pred_y, y, reduction="mean")
        mse_losss = F.mse_loss(pred_y, y, reduction="mean")
        moe_loss = output_dict.get(
            "moe_loss", torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)
        )
        total_loss = MAE + moe_loss
        return {
            "total_loss": total_loss,
            "mae_loss": MAE,
            "mse_loss": mse_losss,
            "moe_loss": moe_loss,
        }

    def get_batch_gates(self) -> Optional[torch.Tensor]:
        """Get the gating weights from the most recent forward pass.

        Returns:
            Tensor of shape (batch_size, num_experts) or None if forward hasn't been called.
        """
        return self.batch_gates

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute squared coefficient of variation for load balancing.

        This metric encourages uniform distribution across experts by penalizing
        high variance in expert usage.

        Args:
            x: Input tensor (e.g., expert loads or importance weights).

        Returns:
            Squared coefficient of variation scalar.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute the load (number of samples) assigned to each expert.

        Args:
            gates: Gating tensor of shape (batch_size, num_experts).

        Returns:
            Load tensor of shape (num_experts,) with counts of assigned samples.
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_values: torch.Tensor,
        noisy_values: torch.Tensor,
        noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute probability that each value is in top-k after adding noise.

        This is used for load balancing during training by computing differentiable
        probabilities of expert selection.

        Args:
            clean_values: Clean logits of shape (batch, num_experts).
            noisy_values: Noisy logits of shape (batch, num_experts).
            noise_stddev: Noise standard deviations of shape (batch, num_experts).
            noisy_top_values: Top-k noisy values of shape (batch, k+1).

        Returns:
            Probabilities of shape (batch, num_experts).
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(
        self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute noisy top-k gating for expert selection.

        Implements the gating mechanism from "Outrageously Large Neural Networks:
        The Sparsely-Gated Mixture-of-Experts Layer" (https://arxiv.org/abs/1701.06538).
        Adds tunable noise during training for exploration.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            train: Whether model is in training mode (adds noise if True).
            noise_epsilon: Small constant for numerical stability.

        Returns:
            gates: Sparse gating weights of shape (batch_size, num_experts).
            load: Load assigned to each expert of shape (num_experts,).
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
            # Add this safety check to ensure we always have at least one expert selected
        if (logits.sum(dim=1) == 0).any():
            # Add a small positive value to ensure we have non-zero logits
            logits = logits + 1e-5

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=self.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Safety check - ensure at least one expert is selected per sample
        if (gates.sum(dim=1) < 1e-6).any():
            # Force selection of the top expert for samples with no experts
            problematic_samples = (gates.sum(dim=1) < 1e-6).nonzero().squeeze(1)
            if problematic_samples.numel() > 0:  # If there are problematic samples
                # Select the top expert for these samples
                top_expert = top_indices[problematic_samples, 0]
                # Set a minimum value for the gate
                gates[problematic_samples, top_expert] = 0.1

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


def train(
    model: Union[VAE, MoE_VAE],
    train_dl: DataLoader,
    device: torch.device,
    epochs: int = 200,
    optimizer: Optional[torch.optim.Optimizer] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a VAE or MoE-VAE model.

    Args:
        model: VAE or MoE_VAE model to train.
        train_dl: DataLoader providing training batches.
        device: Device to train on (CPU or CUDA).
        epochs: Number of training epochs.
        optimizer: PyTorch optimizer. If None, must be configured externally.
        save_dir: Directory to save the best model checkpoint.

    Returns:
        Dictionary containing:
            - 'total_loss': List of total losses per epoch.
            - 'l1_loss': List of L1 losses per epoch.
            - 'best_loss': Best total loss achieved.
    """
    model.train()
    min_total_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_model_minloss.pth")

    total_list = []
    l1_list = []

    for epoch in range(epochs):
        total_loss_epoch = 0.0
        l1_epoch = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            output_dict = model(x)
            output_dict["y"] = y

            loss_dict = model.loss_fn(output_dict)

            loss = loss_dict["total_loss"]
            l1 = loss_dict["mae_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            l1_epoch += l1.item()

        avg_total_loss = total_loss_epoch / len(train_dl)
        avg_l1 = l1_epoch / len(train_dl)

        print(f"[Epoch {epoch+1}] Total: {avg_total_loss:.4f} | L1: {avg_l1:.4f}")
        total_list.append(avg_total_loss)
        l1_list.append(avg_l1)

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_path)

    return {"total_loss": total_list, "l1_loss": l1_list, "best_loss": min_total_loss}


def evaluate(
    model: Union[VAE, MoE_VAE],
    test_dl: DataLoader,
    device: torch.device,
    TSS_scalers_dict: Optional[Dict[str, Any]] = None,
    log_offset: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a VAE or MoE-VAE model on test data.

    Args:
        model: Trained VAE or MoE_VAE model.
        test_dl: DataLoader providing test batches.
        device: Device to run evaluation on (CPU or CUDA).
        TSS_scalers_dict: Optional dictionary with 'log' and 'robust' scalers
            for inverse transformation. If None, uses simple log offset.
        log_offset: Offset for inverse log transformation if TSS_scalers_dict is None.

    Returns:
        predictions_inverse: Predictions in original scale.
        actuals_inverse: Ground truth values in original scale.
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            output_dict = model(x)
            y_pred = output_dict["pred_y"]
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # === Inverse transformation ===
    if TSS_scalers_dict is not None:
        log_scaler = TSS_scalers_dict["log"]
        robust_scaler = TSS_scalers_dict["robust"]

        # First reverse min-max, then reverse log
        predictions_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(predictions, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )

        actuals_inverse = (
            log_scaler.inverse_transform(
                torch.tensor(
                    robust_scaler.inverse_transform(
                        torch.tensor(actuals, dtype=torch.float32)
                    ),
                    dtype=torch.float32,
                )
            )
            .numpy()
            .flatten()
        )
    else:
        predictions_inverse = (10 ** predictions.flatten()) - log_offset
        actuals_inverse = (10 ** actuals.flatten()) - log_offset

    return predictions_inverse, actuals_inverse
