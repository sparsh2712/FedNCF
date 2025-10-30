"""
Communication strategies for federated learning.
Supports different compression techniques: SignSGD, Quantization, Sparsification.
"""

import torch
import numpy as np
from pathlib import Path


class CommunicationStrategy:
    """Base class for communication strategies."""

    def compress(self, model, reference_model=None):
        """
        Compress model for communication.

        Args:
            model: The trained model to compress
            reference_model: Optional reference model (e.g., previous round's model)

        Returns:
            Compressed representation
        """
        raise NotImplementedError

    def decompress(self, compressed_data, reference_model=None):
        """
        Decompress received data back to model state dict.

        Args:
            compressed_data: Compressed model representation
            reference_model: Optional reference model for reconstruction

        Returns:
            Model state dict
        """
        raise NotImplementedError

    def save_compressed(self, compressed_data, path):
        """Save compressed data to disk."""
        raise NotImplementedError

    def load_compressed(self, path):
        """Load compressed data from disk."""
        raise NotImplementedError


class BaselineStrategy(CommunicationStrategy):
    """Baseline: No compression, send full model weights."""

    def compress(self, model, reference_model=None):
        """Return full model state dict."""
        return model.state_dict()

    def decompress(self, compressed_data, reference_model=None):
        """Return state dict as is."""
        return compressed_data

    def save_compressed(self, compressed_data, path):
        """Save full model state dict."""
        torch.save(compressed_data, path)

    def load_compressed(self, path):
        """Load full model state dict."""
        return torch.load(path)


class SignSGDStrategy(CommunicationStrategy):
    """
    SignSGD: Compress gradients/updates to their signs (1 bit per parameter).

    Algorithm:
    1. Compute delta = trained_weights - reference_weights
    2. Compress: signs = sign(delta)
    3. Transmit only signs (1 bit vs 32 bits = 32x compression)
    4. Reconstruct: aggregated_signs via majority voting
    5. Update: reference_weights + learning_rate * aggregated_signs
    """

    def __init__(self, learning_rate=1.0):
        """
        Args:
            learning_rate: Step size for sign-based updates
        """
        self.learning_rate = learning_rate

    def compress(self, model, reference_model):
        """
        Compress model to signs of weight differences.

        Args:
            model: Trained model
            reference_model: Reference model (server model before training)

        Returns:
            Dict of parameter signs (int8: -1, 0, +1)
        """
        if reference_model is None:
            raise ValueError("SignSGD requires a reference model")

        signs = {}
        model_state = model.state_dict()
        ref_state = reference_model.state_dict()

        for key in model_state.keys():
            # Compute difference (gradient-like update)
            delta = model_state[key] - ref_state[key]

            # Extract sign: -1, 0, or +1
            signs[key] = torch.sign(delta).to(torch.int8)

        return signs

    def decompress(self, compressed_data, reference_model):
        """
        Reconstruct model state dict from signs.

        Args:
            compressed_data: Dict of signs
            reference_model: Reference model to apply signs to

        Returns:
            Reconstructed state dict
        """
        if reference_model is None:
            raise ValueError("SignSGD requires a reference model for decompression")

        reconstructed = {}
        ref_state = reference_model.state_dict()

        for key in compressed_data.keys():
            # Convert signs back to float and apply learning rate
            signed_update = compressed_data[key].float() * self.learning_rate

            # Apply update to reference weights
            reconstructed[key] = ref_state[key] + signed_update

        return reconstructed

    def save_compressed(self, compressed_data, path):
        """Save compressed signs (int8 format)."""
        torch.save(compressed_data, path)

    def load_compressed(self, path):
        """Load compressed signs."""
        return torch.load(path)

    def aggregate_signs(self, sign_list):
        """
        Aggregate signs from multiple clients using majority voting.

        Args:
            sign_list: List of sign dicts from different clients

        Returns:
            Aggregated signs (majority vote)
        """
        if not sign_list:
            return {}

        aggregated = {}

        for key in sign_list[0].keys():
            # Stack all client signs for this parameter
            stacked = torch.stack([signs[key].float() for signs in sign_list])

            # Majority voting: sum and take sign
            # Positive sum -> +1, negative sum -> -1, zero sum -> 0
            aggregated[key] = torch.sign(torch.sum(stacked, dim=0)).to(torch.int8)

        return aggregated


class QuantizationStrategy(CommunicationStrategy):
    """
    Gradient Quantization: Reduce precision from float32 (32-bit) to int8 (8-bit).
    Uses symmetric quantization around zero for simplicity.

    Compression: 4x (32 bits → 8 bits per parameter)
    """

    def __init__(self, num_bits=8):
        """
        Args:
            num_bits: Number of bits for quantization (default: 8)
        """
        self.num_bits = num_bits
        if num_bits != 8:
            raise NotImplementedError("Only 8-bit quantization is currently supported")

        # For int8: range is [-128, 127]
        self.qmin = -128
        self.qmax = 127

    def compress(self, model, reference_model=None):
        """
        Quantize model weights to int8.

        Args:
            model: Model to quantize
            reference_model: Not used for quantization

        Returns:
            Dict with 'quantized' (int8 tensors) and 'scales' (float32 per tensor)
        """
        quantized = {}
        scales = {}

        for key, param in model.state_dict().items():
            # Compute scale: max absolute value divided by 127
            # This ensures symmetric quantization around zero
            max_abs = torch.max(torch.abs(param))

            if max_abs == 0:
                # Handle all-zero tensors
                scale = 1.0
            else:
                scale = max_abs / 127.0

            # Quantize: divide by scale, round, and clamp to [-128, 127]
            quantized_param = torch.clamp(
                torch.round(param / scale),
                self.qmin,
                self.qmax
            ).to(torch.int8)

            quantized[key] = quantized_param
            scales[key] = scale

        return {'quantized': quantized, 'scales': scales}

    def decompress(self, compressed_data, reference_model=None):
        """
        Dequantize int8 back to float32.

        Args:
            compressed_data: Dict with 'quantized' and 'scales'
            reference_model: Not used

        Returns:
            State dict with float32 tensors
        """
        state_dict = {}

        for key in compressed_data['quantized'].keys():
            quantized_param = compressed_data['quantized'][key]
            scale = compressed_data['scales'][key]

            # Dequantize: multiply by scale
            dequantized = quantized_param.float() * scale
            state_dict[key] = dequantized

        return state_dict

    def save_compressed(self, compressed_data, path):
        """Save quantized data (int8 tensors + scales)."""
        torch.save(compressed_data, path)

    def load_compressed(self, path):
        """Load quantized data."""
        return torch.load(path)


class SparsificationStrategy(CommunicationStrategy):
    """
    Gradient Sparsification: Send only top-k% of gradients by magnitude.

    Algorithm:
    1. Compute gradient/delta = trained_weights - reference_weights
    2. Select top-k% by absolute magnitude
    3. Transmit only: (indices, values) of top-k elements
    4. Rest are implicitly zero

    Compression ratio depends on sparsity (e.g., 10% → ~10x compression)
    """

    def __init__(self, sparsity=0.9):
        """
        Args:
            sparsity: Fraction of gradients to zero out (0.9 = keep only 10%)
        """
        self.sparsity = sparsity
        self.keep_ratio = 1.0 - sparsity

    def compress(self, model, reference_model):
        """
        Compress model by keeping only top-k% of weight differences.

        Args:
            model: Trained model
            reference_model: Reference model (needed to compute delta)

        Returns:
            Dict with sparse representation: {'indices': ..., 'values': ..., 'shapes': ...}
        """
        if reference_model is None:
            raise ValueError("Sparsification requires a reference model")

        compressed = {
            'indices': {},
            'values': {},
            'shapes': {},
            'keys': list(model.state_dict().keys())
        }

        model_state = model.state_dict()
        ref_state = reference_model.state_dict()

        for key in model_state.keys():
            # Compute gradient/delta
            delta = model_state[key] - ref_state[key]
            original_shape = delta.shape

            # Flatten for easier indexing
            delta_flat = delta.flatten()
            num_params = delta_flat.numel()
            k = max(1, int(num_params * self.keep_ratio))  # Keep at least 1

            # Get top-k indices by absolute magnitude
            abs_delta = torch.abs(delta_flat)
            _, top_k_indices = torch.topk(abs_delta, k)

            # Extract top-k values
            top_k_values = delta_flat[top_k_indices]

            # Store sparse representation
            compressed['indices'][key] = top_k_indices.to(torch.int32)  # int32 for indices
            compressed['values'][key] = top_k_values.to(torch.float32)  # float32 for values
            compressed['shapes'][key] = original_shape

        return compressed

    def decompress(self, compressed_data, reference_model):
        """
        Reconstruct full model from sparse representation.

        Args:
            compressed_data: Sparse dict with indices, values, shapes
            reference_model: Reference model to apply updates to

        Returns:
            Full state dict
        """
        if reference_model is None:
            raise ValueError("Sparsification requires a reference model for decompression")

        state_dict = {}
        ref_state = reference_model.state_dict()

        for key in compressed_data['keys']:
            indices = compressed_data['indices'][key]
            values = compressed_data['values'][key]
            shape = compressed_data['shapes'][key]

            # Reconstruct sparse gradient
            num_params = torch.prod(torch.tensor(shape)).item()
            sparse_delta = torch.zeros(num_params)
            sparse_delta[indices.long()] = values

            # Reshape back to original shape
            sparse_delta = sparse_delta.reshape(shape)

            # Apply to reference weights
            state_dict[key] = ref_state[key] + sparse_delta

        return state_dict

    def save_compressed(self, compressed_data, path):
        """Save sparse representation."""
        torch.save(compressed_data, path)

    def load_compressed(self, path):
        """Load sparse representation."""
        return torch.load(path)


class SparsificationMemoryStrategy(CommunicationStrategy):
    """
    Gradient Sparsification with Error Feedback (Memory).

    Algorithm:
    1. Compute gradient = trained_weights - reference_weights
    2. Add accumulated memory: gradient_corrected = gradient + memory
    3. Select top-k% of gradient_corrected
    4. Store residual: memory = gradient_corrected - sparse_gradient
    5. Transmit sparse_gradient
    6. Next round: residual accumulates until transmitted

    Benefits: No information loss, better convergence than vanilla sparsification
    """

    def __init__(self, sparsity=0.9):
        """
        Args:
            sparsity: Fraction of gradients to zero out (0.9 = keep only 10%)
        """
        self.sparsity = sparsity
        self.keep_ratio = 1.0 - sparsity
        self.memory = {}  # Store residual per parameter

    def compress(self, model, reference_model):
        """
        Compress with error feedback.

        Args:
            model: Trained model
            reference_model: Reference model

        Returns:
            Sparse representation
        """
        if reference_model is None:
            raise ValueError("Sparsification requires a reference model")

        compressed = {
            'indices': {},
            'values': {},
            'shapes': {},
            'keys': list(model.state_dict().keys())
        }

        model_state = model.state_dict()
        ref_state = reference_model.state_dict()

        for key in model_state.keys():
            # Compute gradient/delta
            delta = model_state[key] - ref_state[key]
            original_shape = delta.shape

            # Initialize memory if first time
            if key not in self.memory:
                self.memory[key] = torch.zeros_like(delta)

            # Add accumulated error from previous rounds
            delta_corrected = delta + self.memory[key]

            # Flatten for easier indexing
            delta_flat = delta_corrected.flatten()
            num_params = delta_flat.numel()
            k = max(1, int(num_params * self.keep_ratio))

            # Get top-k indices by absolute magnitude
            abs_delta = torch.abs(delta_flat)
            _, top_k_indices = torch.topk(abs_delta, k)

            # Extract top-k values
            top_k_values = delta_flat[top_k_indices]

            # Reconstruct sparse version
            sparse_delta_flat = torch.zeros_like(delta_flat)
            sparse_delta_flat[top_k_indices] = top_k_values
            sparse_delta = sparse_delta_flat.reshape(original_shape)

            # Update memory: store what wasn't transmitted
            self.memory[key] = delta_corrected - sparse_delta

            # Store sparse representation
            compressed['indices'][key] = top_k_indices.to(torch.int32)
            compressed['values'][key] = top_k_values.to(torch.float32)
            compressed['shapes'][key] = original_shape

        return compressed

    def decompress(self, compressed_data, reference_model):
        """
        Reconstruct full model from sparse representation.
        Same as vanilla sparsification.
        """
        if reference_model is None:
            raise ValueError("Sparsification requires a reference model")

        state_dict = {}
        ref_state = reference_model.state_dict()

        for key in compressed_data['keys']:
            indices = compressed_data['indices'][key]
            values = compressed_data['values'][key]
            shape = compressed_data['shapes'][key]

            # Reconstruct sparse gradient
            num_params = torch.prod(torch.tensor(shape)).item()
            sparse_delta = torch.zeros(num_params)
            sparse_delta[indices.long()] = values

            # Reshape back to original shape
            sparse_delta = sparse_delta.reshape(shape)

            # Apply to reference weights
            state_dict[key] = ref_state[key] + sparse_delta

        return state_dict

    def save_compressed(self, compressed_data, path):
        """Save sparse representation."""
        torch.save(compressed_data, path)

    def load_compressed(self, path):
        """Load sparse representation."""
        return torch.load(path)

    def reset_memory(self):
        """Reset accumulated memory (call at start of new experiment)."""
        self.memory = {}


def get_communication_strategy(method, **kwargs):
    """
    Factory function to get communication strategy.

    Args:
        method: One of "baseline", "signsgd", "quantization", "sparsification", "sparsification_memory"
        **kwargs: Additional arguments for the strategy

    Returns:
        CommunicationStrategy instance
    """
    strategies = {
        "baseline": BaselineStrategy,
        "signsgd": SignSGDStrategy,
        "quantization": QuantizationStrategy,
        "sparsification": SparsificationStrategy,
        "sparsification_memory": SparsificationMemoryStrategy
    }

    if method not in strategies:
        raise ValueError(f"Unknown communication method: {method}")

    return strategies[method](**kwargs)
