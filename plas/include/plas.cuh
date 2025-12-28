#pragma once

#include "permuters.cuh"

template <typename T> T* get_3d_tensor_device_memory(int H, int W, int D) {
    T* tensor;
    size_t size = H * W * D * sizeof(T);
    cudaMalloc(&tensor, size);
    return tensor;
}

template <typename T> T* get_2d_tensor_device_memory(int H, int W) {
    T* tensor;
    size_t size = H * W * sizeof(T);
    cudaMalloc(&tensor, size);
    return tensor;
}

template <typename T> T* get_1d_tensor_device_memory(int size) {
    T* tensor;
    size_t size_bytes = size * sizeof(T);
    cudaMalloc(&tensor, size_bytes);
    return tensor;
}

void blockify(const float* grid, float* blockified_grid,
    const float* target_grid, float* blockified_target_grid,
    const int* index, int* blockified_index, int H, int W, int C,
    int block_len_x, int block_len_y, int tx, int ty);

void unblockify(float* grid, const float* blockified_grid, float* target_grid,
    const float* blockified_target_grid, int* index,
    const int* blockified_index, int H, int W, int C,
    int block_len_x, int block_len_y, int tx, int ty);

void group(float* blockified_grid[2], float* blockified_target_grid[2],
    int* blockified_index[2], int* permutation, int block_size,
    int num_blocks, int C, int turn);

void swap_within_group(float* blockified_grid, float* blockified_target_grid,
    float* dist_to_target, int* blockified_index, int C, int num_groups);

// API for the python wrapper
void sort_with_plas(
    const float* grid_input, float* grid_output, int* index_output, float* grid_target, int H, int W, int C, 
    int64_t seed, RandomPermuter* permuter, int min_block_side, int min_filter_side_length,
    float filter_decrease_factor, float improvement_break,
    int min_group_configs, int max_group_configs,
    bool verbose);

// C++ API
std::pair<float*, int*> sort_with_plas(
    const float* grid_input,
    int H, int W, int C, int64_t seed = 1337, float* grid_target = nullptr, RandomPermuter* permuter = new LCGPermuter(), int min_block_side = 4, int min_filter_side_length = 3,
    float filter_decrease_factor = 0.9, float improvement_break = 1e-5,
    int min_group_configs = 1, int max_group_configs = 10,
    bool verbose = false);
