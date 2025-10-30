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
    Gradient Quantization: Reduce precision of weights/gradients.
    (Placeholder for future implementation)
    """

    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def compress(self, model, reference_model=None):
        raise NotImplementedError("Quantization not yet implemented")

    def decompress(self, compressed_data, reference_model=None):
        raise NotImplementedError("Quantization not yet implemented")

    def save_compressed(self, compressed_data, path):
        raise NotImplementedError("Quantization not yet implemented")

    def load_compressed(self, path):
        raise NotImplementedError("Quantization not yet implemented")


class SparsificationStrategy(CommunicationStrategy):
    """
    Gradient Sparsification: Send only top-k% of gradients by magnitude.
    (Placeholder for future implementation)
    """

    def __init__(self, sparsity=0.9):
        """
        Args:
            sparsity: Fraction of gradients to zero out (0.9 = keep only 10%)
        """
        self.sparsity = sparsity

    def compress(self, model, reference_model=None):
        raise NotImplementedError("Sparsification not yet implemented")

    def decompress(self, compressed_data, reference_model=None):
        raise NotImplementedError("Sparsification not yet implemented")

    def save_compressed(self, compressed_data, path):
        raise NotImplementedError("Sparsification not yet implemented")

    def load_compressed(self, path):
        raise NotImplementedError("Sparsification not yet implemented")


def get_communication_strategy(method, **kwargs):
    strategies = {
        "baseline": BaselineStrategy,
        "signsgd": SignSGDStrategy,
        "quantization": QuantizationStrategy,
        "sparsification": SparsificationStrategy
    }

    if method not in strategies:
        raise ValueError(f"Unknown communication method: {method}")

    return strategies[method](**kwargs)
