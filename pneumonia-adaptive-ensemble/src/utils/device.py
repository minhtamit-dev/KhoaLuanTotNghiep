"""Device selection and management utility."""
from typing import Any, Dict
import torch


def get_device(preference: str = "cuda") -> torch.device:
    """Get PyTorch device based on preference and hardware availability.

    Args:
        preference: Desired device string ("cuda", "mps", "cpu").

    Returns:
        torch.device: Available PyTorch device.
    """
    preference = preference.lower()
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors/models to the target device.

    Args:
        data: Tensor, Model, Dict, or List of Tensors.
        device: Target device.

    Returns:
        Data placed on the target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif hasattr(data, "to"):
        return data.to(device)
    return data


def get_device_info() -> Dict[str, Any]:
    """Retrieve diagnostic information about available computing devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
    }
    return info
