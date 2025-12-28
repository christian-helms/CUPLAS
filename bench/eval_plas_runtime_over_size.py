import torch
import numpy as np
import time
import json

from plas import sort_with_plas as cuplas
from plas.legacy_plas import sort_with_plas as legacy_plas
from bench_helpers import avg_L2_dist_between_neighbors


def generate_random_colors(nx, ny, seed=1337):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (nx, ny, 3), dtype=np.int32)


def bench_plas_scaling(sorting_method: str):
    execution_times = []
    quality_scores = []

    sorting_function = (
        cuplas
        if sorting_method == "cuplas"
        else legacy_plas
    )

    for p in range(4, 12):
        H = 2**p
        W = 2**p
        C = 3
        grid = torch.from_numpy(generate_random_colors(H, W).astype(np.float32)).to(
            device="cuda"
        )
        if sorting_method == "legacy_plas":
            grid = grid.permute(2, 0, 1)
        start = time.time()
        sorted_grid, sorted_indices = sorting_function(grid)
        end = time.time()
        if sorting_method == "legacy_plas":
            sorted_grid = sorted_grid.permute(1, 2, 0)

        execution_times.append({"H": H, "W": W, "C": C, "time": end - start})
        quality_scores.append(
            {"H": H, "W": W, "C": C, "score": avg_L2_dist_between_neighbors(sorted_grid)}
        )

        print(f"Time taken for {H}x{W}x{C} grid: {round(end - start, 3)} seconds")

        # cv2.imshow("sorted_grid", (sorted_grid / 255.0).cpu().numpy())
        # cv2.waitKey(0)

    with open(f"bench/{sorting_method}_runtime_over_size.json", "w") as f:
        json.dump(execution_times, f)

    with open(f"bench/{sorting_method}_quality_over_size.json", "w") as f:
        json.dump(quality_scores, f)


if __name__ == "__main__":
    bench_plas_scaling("cuplas")
    bench_plas_scaling("legacy_plas")
