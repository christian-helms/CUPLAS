import torch
from torch import Tensor
from . import cuplas  # noqa # type: ignore

### ------ Wrappers for custom torch CUDA/C++ operators ------ ###


def random_philox_permutation(
    n: int, num_rounds: int, dummy: Tensor | None = None
) -> Tensor:
    # Choose device without allocating at import time
    if dummy is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dummy = torch.empty(0, device=device)
    if dummy.is_cuda:
        return cuplas.random_philox_permutation_cuda(n, num_rounds, dummy)  # type: ignore
    else:
        return cuplas.random_philox_permutation_cpu(n, num_rounds, dummy)  # type: ignore


def sort_with_plas(
    grid: Tensor,
    grid_target: Tensor = torch.empty(0),
    seed: int = 1337,
    permuter_type: str = "lcg",
    min_block_side: int = 4,
    min_filter_side_length: int = 2,
    filter_decrease_factor: float = 0.9,
    improvement_break: float = 1e-5,
    min_group_configs: int = 3,
    max_group_configs: int = 10,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    grid_output = torch.empty_like(grid)
    index_output = torch.zeros(grid.size(0), grid.size(1), dtype=torch.int32, device=grid.device)
    cuplas.sort_with_plas(  # type: ignore
        grid,
        grid_output,
        index_output,
        grid_target.to(grid.device),
        seed,
        permuter_type,
        min_block_side,
        min_filter_side_length,
        filter_decrease_factor,
        improvement_break,
        min_group_configs,
        max_group_configs,
        verbose,
    )
    return grid_output, index_output
