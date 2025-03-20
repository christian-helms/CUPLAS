from plas import sort_with_plas
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import timeit
import io
from PIL import Image


def bench_sort_with_plas(grid, config: DictConfig = OmegaConf.create()):
    bench_log = OmegaConf.create()

    start_time = timeit.default_timer()
    sorted_grid, sorted_index = sort_with_plas(grid, **config) # type: ignore
    end_time = timeit.default_timer()
    time_taken = end_time - start_time
    bench_log.duration = float(time_taken)

    png_size_sum = 0
    for i in range(int(np.ceil(grid.shape[0] / 3))):
        num_channels = min(3 * (i + 1), grid.shape[0]) - 3 * i
        if num_channels < 3:
            for j in range(num_channels):
                png_size = len(
                    tensor_to_png(sorted_grid[:, :, 3 * i + j : 3 * i + j + 1])
                )
                png_size_sum += png_size
        else:
            png_size = len(
                tensor_to_png(sorted_grid[3 * i : 3 * (i + 1), :, :])
            )
        png_size_sum += png_size
    compression_factor = (np.prod(grid.shape) * grid.element_size()) / png_size_sum
    bench_log.png_compression_factor = float(compression_factor)

    and_unsorted = avg_L2_dist_between_neighbors(grid)
    and_sorted = avg_L2_dist_between_neighbors(sorted_grid)
    bench_log.avg_l2_dist_reduction_factor = float(and_unsorted / and_sorted)

    return bench_log


def tensor_to_png(tensor: torch.Tensor):
    """Convert a PyTorch tensor to PNG bytes.

    Args:
        tensor (torch.Tensor): Input tensor of shape (H, W) or (H, W, C).
                             Values should be in range [0, 1] or [0, 255].
    Returns:
        bytes: PNG image data
    """
    if tensor.dim() == 2:  # HxW
        tensor = tensor.unsqueeze(-1)  # Add channel dimension at end
    elif tensor.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D")

    # Ensure tensor is on CPU and convert to uint8 if needed
    tensor = tensor.detach().cpu()
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).clamp(0, 255)
    tensor = tensor.to(torch.uint8)

    # Convert to PIL Image
    if tensor.size(-1) == 1:  # Grayscale
        img = Image.fromarray(tensor.squeeze(-1).numpy(), mode="L")
    elif tensor.size(-1) == 3:  # RGB
        img = Image.fromarray(tensor.numpy(), mode="RGB")
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(-1)}")

    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def avg_L2_dist_between_neighbors(grid):
    """Calculate average L2 distance between neighboring pixels that share an edge.

    Args:
        grid (torch.Tensor): Input tensor of shape (H, W, C)
    Returns:
        float: Average L2 distance between neighboring pixels
    """
    # Calculate differences between adjacent pixels
    h_diff = grid[1:, :, :] - grid[:-1, :, :]  # vertical differences
    w_diff = grid[:, 1:, :] - grid[:, :-1, :]  # horizontal differences

    # Calculate L2 distances
    h_dist = torch.sqrt((h_diff**2).sum(dim=-1))  # sum over channels
    w_dist = torch.sqrt((w_diff**2).sum(dim=-1))  # sum over channels

    # Sum all distances and divide by number of edges
    total_dist = h_dist.sum() + w_dist.sum()
    num_edges = h_dist.numel() + w_dist.numel()

    return float(total_dist / num_edges)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    return device
