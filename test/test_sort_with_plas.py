import torch
import plas
import cv2

def is_permutation(permutation: torch.Tensor) -> torch.Tensor:
    sorted = torch.sort(permutation)[0]
    expected = torch.arange(permutation.size(0), device=permutation.device)
    return torch.all(sorted == expected)

def is_sorted_according_to_index(grid: torch.Tensor, sorted_grid: torch.Tensor, sorted_index: torch.Tensor) -> torch.Tensor:
    flattened_grid = grid.flatten(start_dim=0, end_dim=1)
    flattened_sorted_index = sorted_index.flatten()
    sorted_according_to_index = flattened_grid[flattened_sorted_index, :]
    flattened_sorted_grid = sorted_grid.flatten(start_dim=0, end_dim=1)
    return torch.all((sorted_according_to_index - flattened_sorted_grid).abs() < 1e-9)

def test_sort_with_plas():
    grid = torch.randn(1024, 1024, 3, device="cuda")
    sorted_grid, sorted_index = plas.sort_with_plas(grid, filter_algo=0)
    assert is_permutation(sorted_index.flatten())
    assert is_sorted_according_to_index(grid, sorted_grid, sorted_index)


if __name__ == "__main__":
    grid = torch.randn(2048, 2048, 3, device="cuda")
    sorted_grid, sorted_index = plas.sort_with_plas(grid, filter_algo=0)
    cv2.imshow("grid", grid.cpu().numpy())
    cv2.imshow("sorted_grid", sorted_grid.cpu().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
