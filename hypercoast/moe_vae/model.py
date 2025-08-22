"""Variational Autoencoder and Mixture of Experts models.

This module implements VAE and MoE-VAE architectures for remote sensing data
analysis, including sparse gating mechanisms and training utilities.
"""

import os
import numpy as np

try:
    from pytorch_lightning import LightningModule
    import torch.nn as nn
    import torch
    import torch.nn.functional as F
    from torch.distributions.normal import Normal
except ImportError:
    print("Please install pytorch-lightning")


class VAE(LightningModule):
    """Variational Autoencoder implementation using PyTorch Lightning.

    A standard VAE architecture with configurable encoder/decoder networks,
    support for various activation functions, normalization layers, and
    dropout regularization.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output/reconstructed data.
        latent_dim (int): Dimension of latent space.
        encoder_hidden_dims (list): List of hidden layer dimensions for encoder.
        decoder_hidden_dims (list): List of hidden layer dimensions for decoder.
        activation (str, optional): Activation function name. Supports 'relu',
            'tanh', 'sigmoid', 'leakyrelu'. Defaults to 'leakyrelu'.
        use_norm (str or bool, optional): Normalization type. Can be 'batch',
            'layer', or False. Defaults to False.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        use_softplus_output (bool, optional): Whether to apply softplus to output.
            Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        activation="leakyrelu",
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        """
        Initialize the VAE model.

        Args:
            input_dim (int): Dimension of input data.
            output_dim (int): Dimension of output/reconstructed data.
            latent_dim (int): Dimension of latent space.
            encoder_hidden_dims (list): List of hidden layer dimensions for encoder.
            decoder_hidden_dims (list): List of hidden layer dimensions for decoder.
            activation (str, optional): Activation function name. Supports 'relu',
                'tanh', 'sigmoid', 'leakyrelu'. Defaults to 'leakyrelu'.
            use_norm (str or bool, optional): Normalization type. Can be 'batch',
                'layer', or False. Defaults to False.
            use_dropout (bool, optional): Whether to use dropout. Defaults to False.
            use_softplus_output (bool, optional): Whether to apply softplus to output.
                Defaults to False.
            **kwargs: Additional keyword arguments.
        """
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

    def build_layers(self, input_dim, hidden_dims, use_norm, use_dropout=False):
        """Build sequential neural network layers.

        Args:
            input_dim (int): Input dimension for the first layer.
            hidden_dims (list): List of hidden layer dimensions.
            use_norm (str or bool): Normalization type ('batch', 'layer', or False).
            use_dropout (bool, optional): Whether to include dropout layers.
                Defaults to False.

        Returns:
            nn.Sequential: Sequential container of network layers.
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
            layers.append(self.activation)
            if use_dropout:
                layers.append(nn.Dropout(0.1))
            current_size = next_size
        return nn.Sequential(*layers)

    def encode(self, x):
        """Encode input to latent space parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): Mean of latent distribution.
                - log_var (torch.Tensor): Log variance of latent distribution.
        """
        x = self.encoder_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from latent distribution.

        Args:
            mu (torch.Tensor): Mean of latent distribution.
            log_var (torch.Tensor): Log variance of latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent representation to output space.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        return self.decoder_layers(z)

    def forward(self, x):
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing:
                - 'pred_y': Reconstructed output
                - 'mu': Mean of latent distribution
                - 'log_var': Log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred_y = self.decode(z)
        return {"pred_y": pred_y, "mu": mu, "log_var": log_var}

    def loss_fn(self, output_dict, kld_weight=0.0):
        """Compute VAE loss (reconstruction + KL divergence).

        Args:
            output_dict (dict): Dictionary containing model outputs and targets.
            kld_weight (float, optional): Weight for KL divergence term.
                Defaults to 0.0.

        Returns:
            dict: Dictionary containing different loss components:
                - 'total_loss': Combined loss (MAE + weighted KLD)
                - 'mae_loss': Mean Absolute Error
                - 'mse_loss': Mean Squared Error
                - 'kld_loss': KL Divergence loss
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
    """Helper for implementing a mixture of experts with sparse gating.

    This class handles the distribution of inputs to experts and combines their
    outputs based on gating weights. It optimizes computation by only processing
    inputs for experts with non-zero gates.

    The class provides two main functions:
    - dispatch: Creates input batches for each expert based on gating weights
    - combine: Combines expert outputs weighted by their respective gates

    Args:
        num_experts (int): Number of expert models.
        gates (torch.Tensor): Gating weights of shape [batch_size, num_experts].
            Element [b, e] represents the weight for sending batch element b
            to expert e.

    Example:
        >>> gates = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.0, 0.9]])
        >>> dispatcher = SparseDispatcher(3, gates)
        >>> expert_inputs = dispatcher.dispatch(inputs)
        >>> expert_outputs = [experts[i](expert_inputs[i]) for i in range(3)]
        >>> combined_output = dispatcher.combine(expert_outputs)

    Note:
        Input and output tensors are expected to be 2D [batch, depth]. Caller
        is responsible for reshaping higher-dimensional tensors before dispatch
        and after combine operations.
    """

    def __init__(self, num_experts, gates):
        """Initialize the SparseDispatcher.

        Args:
            num_experts (int): Number of expert models.
            gates (torch.Tensor): Gating weights of shape [batch_size, num_experts].
        """
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

    def dispatch(self, inp):
        """Distribute input tensor to experts based on gating weights.

        Creates separate input tensors for each expert containing only the
        samples assigned to that expert (where gates[b, i] > 0).

        Args:
            inp (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            list: List of tensors, one for each expert. Each tensor contains
                only the inputs assigned to that expert.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Combine expert outputs weighted by gating values.

        Aggregates outputs from all experts for each batch element, weighted
        by the corresponding gate values. The final output for batch element b
        is the sum of expert outputs weighted by gates[b, i].

        Args:
            expert_out (list): List of expert output tensors, each with shape
                [expert_batch_size_i, output_dim].
            multiply_by_gates (bool, optional): Whether to weight outputs by
                gate values. If False, outputs are simply summed. Defaults to True.

        Returns:
            torch.Tensor: Combined output tensor of shape [batch_size, output_dim].
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

    def expert_to_gates(self):
        """Extract gate values for each expert's assigned samples.

        Returns:
            list: List of 1D tensors, one for each expert, containing the
                gate values for samples assigned to that expert.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_VAE(LightningModule):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_dim: integer - size of the input
    output_dim: integer - size of the input
    num_experts: an integer - number of experts
    hidden_dims: an integer - hidden_dims size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        num_experts,
        k=4,
        activation="leakyrelu",
        noisy_gating=True,
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        """
        Initialize the MoE-VAE model.

        Args:
            input_dim (int): Dimension of input data.
            output_dim (int): Dimension of output/reconstructed data.
            latent_dim (int): Dimension of latent space.
            encoder_hidden_dims (list): List of hidden layer dimensions for encoder.
            decoder_hidden_dims (list): List of hidden layer dimensions for decoder.
            num_experts (int): Number of experts.
            k (int, optional): Number of experts to use for each batch element.
            activation (str, optional): Activation function name.
            noisy_gating (bool, optional): Whether to use noisy gating.
            use_norm (str or bool, optional): Normalization type. Can be 'batch',
                'layer', or False. Defaults to False.
            use_dropout (bool, optional): Whether to use dropout. Defaults to False.
            use_softplus_output (bool, optional): Whether to apply softplus to output.
                Defaults to False.
            **kwargs: Additional keyword arguments.
        """
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

    def forward(self, x, moe_weight=1e-2):
        """
        Forward pass of the MoE-VAE model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            moe_weight (float, optional): Multiplier for load-balancing loss.
                Defaults to 1e-2.

        Returns:
            dict: Dictionary containing:
                - 'pred_y': Predicted output tensor of shape [batch_size, output_dim].
                - 'moe_loss': Load-balancing loss.
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

    def loss_fn(self, output_dict) -> torch.Tensor:
        """
        Compute loss between model output and target.

        Args:
            output: Model output tensor of shape (batch, output_dim)
            target: Target tensor of shape (batch, output_dim)

        Returns:
            loss: Scalar tensor representing the loss
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

    def get_batch_gates(self):
        """Get the gating weights from the last forward pass.

        Returns:
            torch.Tensor: Gating weights of shape [batch_size, num_experts].
        """
        return self.batch_gates

    def cv_squared(self, x):
        """Compute squared coefficient of variation for load balancing.

        Calculates the squared coefficient of variation (variance/mean²) which
        serves as a loss term to encourage uniform distribution across experts.

        Args:
            x (torch.Tensor): Input tensor (typically expert loads or importance).

        Returns:
            torch.Tensor: Scalar tensor representing squared coefficient of variation.
                Returns 0 for single-element tensors.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Convert gate weights to expert load counts.

        Computes the number of examples assigned to each expert (with gate > 0).

        Args:
            gates (torch.Tensor): Gate weights of shape [batch_size, num_experts].

        Returns:
            torch.Tensor: Load count per expert of shape [num_experts].
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Compute probability of expert being in top-k selection.

        Helper function for noisy top-k gating that computes the probability
        each expert would be selected given different noise realizations.
        This enables differentiable load balancing.

        Args:
            clean_values (torch.Tensor): Clean logits of shape [batch, num_experts].
            noisy_values (torch.Tensor): Noisy logits of shape [batch, num_experts].
            noise_stddev (torch.Tensor): Noise standard deviation of same shape.
            noisy_top_values (torch.Tensor): Top-k+1 noisy values for thresholding.

        Returns:
            torch.Tensor: Probability of each expert being in top-k,
                shape [batch, num_experts].
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

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating mechanism for expert selection.

        Implements the noisy top-k gating from "Outrageously Large Neural Networks"
        (https://arxiv.org/abs/1701.06538). Adds controlled noise during training
        to improve load balancing across experts.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, input_dim].
            train (bool): Whether model is in training mode (adds noise if True).
            noise_epsilon (float, optional): Minimum noise standard deviation.
                Defaults to 1e-2.

        Returns:
            tuple: A tuple containing:
                - gates (torch.Tensor): Sparse gate weights [batch_size, num_experts]
                - load (torch.Tensor): Expert load for balancing [num_experts]
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


class MoE_VAE_Token(LightningModule):
    """Token-wise Mixture of Experts VAE for spectral data analysis.

    This variant of MoE-VAE divides the input spectral bands among different
    experts, with each expert processing a specific spectral segment. This is
    particularly useful for hyperspectral data where different spectral regions
    may have distinct characteristics.

    Args:
        input_dim (int): Total dimension of input spectral data.
        output_dim (int): Dimension of output data.
        latent_dim (int): Dimension of latent space for each VAE expert.
        encoder_hidden_dims (list): Hidden layer dimensions for encoder networks.
        decoder_hidden_dims (list): Hidden layer dimensions for decoder networks.
        num_experts (int): Number of expert VAE models (spectral segments).
        k (int, optional): Kept for compatibility, unused in token-wise mode.
            Defaults to 4.
        activation (str, optional): Activation function name. Defaults to 'leakyrelu'.
        noisy_gating (bool, optional): Kept for compatibility, unused in token-wise mode.
            Defaults to True.
        use_norm (str or bool, optional): Normalization type. Defaults to False.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        use_softplus_output (bool, optional): Whether to apply softplus to output.
            Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        encoder_hidden_dims,
        decoder_hidden_dims,
        num_experts,
        k=4,
        activation="leakyrelu",
        noisy_gating=True,
        use_norm=False,
        use_dropout=False,
        use_softplus_output=False,
        **kwargs,
    ):
        super(MoE_VAE_Token, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.activation = activation
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_softplus_output = use_softplus_output

        # instantiate experts
        self.sub_input_dims = [input_dim // num_experts] * (num_experts - 1)
        self.sub_input_dims.append(input_dim - sum(self.sub_input_dims))

        self.experts = nn.ModuleList(
            [
                VAE(
                    sub_dim,
                    sub_dim,
                    self.latent_dim,
                    self.encoder_hidden_dims,
                    self.decoder_hidden_dims,
                    self.activation,
                    use_norm=self.use_norm,
                    use_dropout=self.use_dropout,
                    use_softplus_output=self.use_softplus_output,
                )
                for sub_dim in self.sub_input_dims
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
        self.k = k
        assert self.k <= self.num_experts

    def forward(self, x, moe_weight=0.0):
        """
        Token-wise MoE forward pass:
        Each expert processes a different spectral segment of the input.

        Args:
            x: Tensor of shape [batch_size, input_dim]
            moe_weight: kept for compatibility, but unused in token-wise mode.

        Returns:
            A dict with:
                'pred_y': reconstructed tensor of shape [batch_size, input_dim]
                'moe_loss': dummy 0.0 (no gating loss in token-wise)
        """
        # Split the input into band segments for each expert
        x_chunks = torch.split(x, self.sub_input_dims, dim=1)

        expert_outputs = []
        for i in range(self.num_experts):
            out_i = self.experts[i](x_chunks[i])["pred_y"]
            expert_outputs.append(out_i)

        pred_y = torch.cat(expert_outputs, dim=1)
        return {
            "pred_y": pred_y,
            "moe_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
        }

    def loss_fn(self, output_dict):
        """Compute loss for token-wise MoE-VAE model.

        Computes reconstruction loss without MoE-specific penalties since
        no gating mechanism is used in the token-wise approach.

        Args:
            output_dict (dict): Dictionary containing model outputs and targets:
                - 'pred_y': Model predictions
                - 'y': Target values
                - 'moe_loss': Always zero for token-wise model

        Returns:
            dict: Dictionary containing loss components:
                - 'total_loss': MAE loss (no MoE penalty)
                - 'mae_loss': Mean Absolute Error
                - 'mse_loss': Mean Squared Error
                - 'moe_loss': Zero tensor
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


def train(model, train_dl, device, epochs=200, optimizer=None, save_dir=None):
    """Train the MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        train_dl (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device to use for training.
        epochs (int, optional): Number of epochs to train. Defaults to 200.
        optimizer (torch.optim.Optimizer, optional): Optimizer to use for training.
        save_dir (str, optional): Directory to save the model. Defaults to None.

    Returns:
        dict: Dictionary containing training metrics:
            - 'total_loss': List of total loss values per epoch.
            - 'l1_loss': List of L1 loss values per epoch.
            - 'best_loss': Minimum total loss value.
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


def evaluate(model, test_dl, device, TSS_scalers_dict=None, log_offset=0.01):
    """Evaluate the MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_dl (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to use for evaluation.
        TSS_scalers_dict (dict, optional): Dictionary containing scalers for TSS.
        log_offset (float, optional): Log offset for predictions. Defaults to 0.01.

    Returns:
        tuple: Tuple containing:
            - predictions_inverse (numpy.ndarray): Inverse transformed predictions.
            - actuals_inverse (numpy.ndarray): Inverse transformed actuals.
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


def evaluate_token(model, test_dl, device, TSS_scalers_dict=None, log_offset=0.01):
    """Evaluate the token-wise MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_dl (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to use for evaluation.
        TSS_scalers_dict (dict, optional): Dictionary containing scalers for TSS.
        log_offset (float, optional): Log offset for predictions. Defaults to 0.01.

    Returns:
        tuple: Tuple containing:
            - predictions_inverse (numpy.ndarray): Inverse transformed predictions.
            - actuals_inverse (numpy.ndarray): Inverse transformed actuals.
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            output_dict = model(x)
            y_pred = output_dict["pred_y"]  # [B, token_len]

            if y_pred.ndim == 2:
                y_pred = y_pred.mean(dim=1, keepdim=True)  # [B, 1]

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
