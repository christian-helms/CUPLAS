from plas import sort_with_plas as cuplas
from plas.legacy_plas import sort_with_plas as legacy_plas
from bench_helpers import avg_L2_dist_between_neighbors

import torch
import time
import numpy as np

rng = np.random.default_rng(1337)

grid = rng.integers(0, 256, (512, 512, 3), dtype=np.int32)
grid = torch.from_numpy(grid.astype(np.float32)).to(device="cuda")

sorting_methods = ["cuplas", "legacy_plas"]

for sorting_method in sorting_methods:
    if sorting_method == "legacy_plas":
        grid_to_be_sorted = grid.clone().permute(2, 0, 1)
        sorting_function = legacy_plas
    else:
        grid_to_be_sorted = grid.clone()
        sorting_function = cuplas

    # record sorting time
    start_time = time.time()
    sorted_grid, sorted_indices = sorting_function(grid_to_be_sorted)
    if sorting_method == "legacy_plas":
        sorted_grid = sorted_grid.permute(1, 2, 0)
    finish_time = time.time()

    print(
        f"Sorting time with {sorting_method}: {round(finish_time - start_time, 3)} seconds"
    )
    print(f"Avg l2 neighbor distance: {avg_L2_dist_between_neighbors(sorted_grid)}")
    # record intermediate grids in teh sorting process
    # and then save it as a video
