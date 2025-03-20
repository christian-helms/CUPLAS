#include "plas.cuh"

void test_blockify() {
    int H = 5;
    int W = 5;
    int C = 1;
    int block_len_x = 2;
    int block_len_y = 2;
    int tx = 1;
    int ty = 1;
    int num_blocks = ((W - tx) / block_len_x) * ((H - ty) / block_len_y);
    int block_size = block_len_x * block_len_y;

    float* host_grid = new float[H * W * C];
    float* host_target_grid = new float[H * W * C];
    int* host_index = new int[H * W];

    for (int i = 0; i < H * W * C; i++) {
        host_grid[i] = i;
        host_target_grid[i] = i;
        host_index[i] = i;
    }

    float* grid = get_3d_tensor_device_memory<float>(H, W, C);
    float* target_grid = get_3d_tensor_device_memory<float>(H, W, C);
    int* index = get_2d_tensor_device_memory<int>(H, W);
    cudaMemcpy(grid, host_grid, H * W * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target_grid, host_target_grid, H * W * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(index, host_index, H * W * sizeof(int), cudaMemcpyHostToDevice);

    int* blockified_index = get_1d_tensor_device_memory<int>(num_blocks * block_size);
    float* blockified_grid = get_2d_tensor_device_memory<float>(num_blocks * block_size, C);
    float* blockified_target_grid = get_2d_tensor_device_memory<float>(num_blocks * block_size, C);

    blockify(grid, blockified_grid, target_grid, blockified_target_grid, index, blockified_index, H, W, C, block_len_x, block_len_y, tx, ty);

    float* host_blockified_grid = new float[num_blocks * block_size * C];
    float* host_blockified_target_grid = new float[num_blocks * block_size * C];
    int* host_blockified_index = new int[num_blocks * block_size];
    cudaMemcpy(host_blockified_grid, blockified_grid, num_blocks * block_size * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_blockified_target_grid, blockified_target_grid, num_blocks * block_size * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_blockified_index, blockified_index, num_blocks * block_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\nBlockified grid: [";
    for (int i = 0; i < num_blocks * block_size * C; i++) {
        std::cout << host_blockified_grid[i];
        if (i < num_blocks * block_size * C - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Blockified target grid: [";
    for (int i = 0; i < num_blocks * block_size * C; i++) {
        std::cout << host_blockified_target_grid[i];
        if (i < num_blocks * block_size * C - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Blockified index: [";
    for (int i = 0; i < num_blocks * block_size; i++) {
        std::cout << host_blockified_index[i];
        if (i < num_blocks * block_size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Clean up
    delete[] host_grid;
    delete[] host_target_grid;
    delete[] host_index;
    delete[] host_blockified_grid;
    delete[] host_blockified_target_grid;
    delete[] host_blockified_index;
    cudaFree(grid);
    cudaFree(target_grid);
    cudaFree(index);
    cudaFree(blockified_grid);
    cudaFree(blockified_target_grid);
    cudaFree(blockified_index);
}

void test_group(int block_size, int num_blocks, int C, int seed, bool verbose = false) {
    int N = num_blocks * block_size;

    float* host_grid = new float[N * C];
    float* host_target_grid = new float[N * C];
    int* host_index = new int[N];
    float* host_grid_after = new float[N * C];
    float* host_target_grid_after = new float[N * C];
    int* host_index_after = new int[N];

    // Initialize sequential values for host_grid and host_index
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < C; c++) {
            host_grid[i * C + c] = i;  // Each channel gets the same value for testing
            host_target_grid[i * C + c] = i;
        }
        host_index[i] = i;
    }

    // Allocate device memory
    float* blockified_grid[2];
    float* blockified_target_grid[2];
    int* blockified_index[2];
    blockified_grid[0] = get_1d_tensor_device_memory<float>(N * C);
    blockified_grid[1] = get_1d_tensor_device_memory<float>(N * C);
    blockified_target_grid[0] = get_1d_tensor_device_memory<float>(N * C);
    blockified_target_grid[1] = get_1d_tensor_device_memory<float>(N * C);
    blockified_index[0] = get_1d_tensor_device_memory<int>(N);
    blockified_index[1] = get_1d_tensor_device_memory<int>(N);

    // Copy data to device
    cudaMemcpy(blockified_grid[0], host_grid, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blockified_target_grid[0], host_target_grid, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blockified_index[0], host_index, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create random permutation for the block_size
    int* permutation = get_1d_tensor_device_memory<int>(block_size);
    int* host_perm = new int[block_size];

    // Initialize sequential permutation
    for (int i = 0; i < block_size; i++) {
        host_perm[i] = i;
    }

    // Fisher-Yates shuffle to create random permutation
    std::mt19937 gen(seed);
    for (int i = block_size - 1; i > 0; i--) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        std::swap(host_perm[i], host_perm[j]);
    }

    if (verbose) {
        std::cout << "Random permutation: ";
        for (int i = 0; i < block_size; i++) {
            std::cout << host_perm[i] << " ";
        }
        std::cout << std::endl;
    }

    cudaMemcpy(permutation, host_perm, block_size * sizeof(int), cudaMemcpyHostToDevice);

    // Apply group function
    bool turn = 0;
    group(blockified_grid, blockified_target_grid, blockified_index, permutation, block_size, num_blocks, C, turn);

    // Copy results back to host
    cudaMemcpy(host_grid_after, blockified_grid[1], N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_target_grid_after, blockified_target_grid[1], N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_index_after, blockified_index[1], N * sizeof(int), cudaMemcpyDeviceToHost);

    if (verbose) {
        std::cout << "Input grid (first channel): ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid[i * C] << " ";
        }
        std::cout << std::endl;

        std::cout << "Output grid (first channel): ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid_after[i * C] << " ";
        }
        std::cout << std::endl;
    }

    // Verify results
    bool test_passed = true;
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_size; i++) {
            int block_start = b * block_size;
            for (int c = 0; c < C; c++) {
                int expected = host_grid[(block_start + host_perm[i]) * C + c];
                if (host_grid_after[(block_start + i) * C + c] != expected) {
                    test_passed = false;
                    if (verbose) {
                        std::cout << "Block " << b << " position " << i << " channel " << c
                            << " expected " << expected
                            << " but got " << host_grid_after[(block_start + i) * C + c] << std::endl;
                    }
                }
            }
        }
    }
    assert(test_passed);

    if (verbose && test_passed) {
        std::cout << "Group test passed!" << std::endl;
    }

    // Cleanup
    delete[] host_grid;
    delete[] host_target_grid;
    delete[] host_index;
    delete[] host_grid_after;
    delete[] host_target_grid_after;
    delete[] host_index_after;
    delete[] host_perm;
    cudaFree(blockified_grid[0]);
    cudaFree(blockified_grid[1]);
    cudaFree(blockified_target_grid[0]);
    cudaFree(blockified_target_grid[1]);
    cudaFree(blockified_index[0]);
    cudaFree(blockified_index[1]);
    cudaFree(permutation);
}

void test_swap_within_group(int num_groups, int C, int seed, bool verbose = false) {
    int N = num_groups * 4;  // Each group has 4 elements

    float* host_grid = new float[N * C];
    float* host_target_grid = new float[N * C];
    float* dist_to_target = get_1d_tensor_device_memory<float>(num_groups);
    int* host_index = new int[N];

    // Initialize sequential values for host_grid and host_index
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < C; c++) {
            host_grid[i * C + c] = i;  // Each channel gets the same value for testing
        }
        host_index[i] = i;
    }

    // Randomly permute elements within each group of 4 in target grid
    std::mt19937 gen(seed);
    for (int g = 0; g < num_groups; g++) {
        int group_start = g * 4;
        // Copy values from host_grid to initialize target grid
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < C; c++) {
                host_target_grid[(group_start + i) * C + c] = host_grid[(group_start + i) * C + c];
            }
        }
        // Fisher-Yates shuffle within the group
        for (int i = 3; i > 0; i--) {
            std::uniform_int_distribution<> dis(0, i);
            int j = dis(gen);
            // Swap all channels together
            for (int c = 0; c < C; c++) {
                std::swap(host_target_grid[(group_start + i) * C + c],
                    host_target_grid[(group_start + j) * C + c]);
            }
        }
    }

    float* grid = get_1d_tensor_device_memory<float>(N * C);
    float* target_grid = get_1d_tensor_device_memory<float>(N * C);
    int* index = get_1d_tensor_device_memory<int>(N);

    cudaMemcpy(grid, host_grid, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target_grid, host_target_grid, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(index, host_index, N * sizeof(int), cudaMemcpyHostToDevice);

    swap_within_group(grid, target_grid, dist_to_target, index, C, num_groups);

    // Copy results back to host for verification
    float* host_grid_after = new float[N * C];
    int* host_index_after = new int[N];
    cudaMemcpy(host_grid_after, grid, N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_index_after, index, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (verbose) {
        std::cout << "Initial grid (first channel): ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid[i * C] << " ";
        }
        std::cout << std::endl;

        std::cout << "Target grid (first channel): ";
        for (int i = 0; i < N; i++) {
            std::cout << host_target_grid[i * C] << " ";
        }
        std::cout << std::endl;

        std::cout << "Final grid (first channel): ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid_after[i * C] << " ";
        }
        std::cout << std::endl;
    }

    // Verify that each group's elements are permuted to minimize distance to target
    bool test_passed = true;
    for (int g = 0; g < num_groups; g++) {
        int group_start = g * 4;

        // For each position in the group, verify all channels match
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < C; c++) {
                float value = host_grid_after[(group_start + i) * C + c];
                float target = host_target_grid[(group_start + i) * C + c];
                if (value != target) {
                    test_passed = false;
                    if (verbose) {
                        std::cout << "Group " << g << " position " << i << " channel " << c
                            << " expected " << target
                            << " but got " << value << std::endl;
                    }
                }
            }
        }
    }
    assert(test_passed);

    if (verbose && test_passed) {
        std::cout << "Swap within group test passed!" << std::endl;
    }

    // Cleanup
    delete[] host_grid;
    delete[] host_target_grid;
    delete[] host_grid_after;
    delete[] host_index;
    delete[] host_index_after;
    cudaFree(grid);
    cudaFree(target_grid);
    cudaFree(index);
    cudaFree(dist_to_target);
}

void test_group_swap_sequence(int block_size, int num_blocks, int seed, bool verbose) {

    int N = num_blocks * block_size;

    float* host_grid = new float[N];
    float* host_target_grid = new float[N];
    int* host_index = new int[N];
    float* host_grid_after = new float[N];
    float* host_target_grid_after = new float[N];
    int* host_index_after = new int[N];

    // Initialize sequential values
    for (int i = 0; i < N; i++) {
        host_grid[i] = i;
        host_target_grid[i] = i;
        host_index[i] = i;
    }

    // Allocate device memory
    float* blockified_grid[2];
    float* blockified_target_grid[2];
    int* blockified_index[2];
    float* dist_to_target = get_1d_tensor_device_memory<float>(num_blocks * block_size / 4);

    blockified_grid[0] = get_1d_tensor_device_memory<float>(N);
    blockified_grid[1] = get_1d_tensor_device_memory<float>(N);
    blockified_target_grid[0] = get_1d_tensor_device_memory<float>(N);
    blockified_target_grid[1] = get_1d_tensor_device_memory<float>(N);
    blockified_index[0] = get_1d_tensor_device_memory<int>(N);
    blockified_index[1] = get_1d_tensor_device_memory<int>(N);

    // Copy data to device
    cudaMemcpy(blockified_grid[0], host_grid, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blockified_target_grid[0], host_target_grid, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blockified_index[0], host_index, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create LCGPermuter
    LCGPermuter permuter;
    permuter.set_seed(1337);

    // Step 1: Group with random permutation
    bool turn = 0;
    int* group_permutation = permuter.get_new_permutation(block_size);

    int* host_group_perm = new int[block_size];
    if (verbose) {
        std::cout << "Group permutation: ";
        cudaMemcpy(host_group_perm, group_permutation, block_size * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < block_size; i++) {
            std::cout << host_group_perm[i] << " ";
        }
        std::cout << std::endl;
    }

    group(blockified_grid, blockified_target_grid, blockified_index,
        group_permutation, block_size, num_blocks, 1, turn);
    turn = 1 - turn;

    // Step 2: Swap within groups (groups of 4)
    swap_within_group(blockified_grid[turn], blockified_target_grid[turn],
        dist_to_target, blockified_index[turn], 1, num_blocks * block_size / 4);

    // Step 3: Degroup using inverse permutation
    if (!permuter.supports_inverse_fusion()) {
        int* inverse_permutation = permuter.get_inverse_permutation(block_size);
        group(blockified_grid, blockified_target_grid, blockified_index,
            inverse_permutation, block_size, num_blocks, 1, turn);
        turn = 1 - turn;
    }

    // Copy results back to host
    cudaMemcpy(host_grid_after, blockified_grid[turn], N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_target_grid_after, blockified_target_grid[turn], N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_index_after, blockified_index[turn], N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print states for debugging
    if (verbose) {
        std::cout << "Initial grid: ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Final grid: ";
        for (int i = 0; i < N; i++) {
            std::cout << host_grid_after[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Final index: ";
        for (int i = 0; i < N; i++) {
            std::cout << host_index_after[i] << " ";
        }
        std::cout << std::endl;

    }
    // Verify that the final permutation is valid
    bool* used = new bool[N]();
    bool permutation_valid = true;
    for (int i = 0; i < N; i++) {
        if (host_index_after[i] < 0 || host_index_after[i] >= N || used[host_index_after[i]]) {
            permutation_valid = false;
            std::cout << "Invalid permutation at position " << i << ": " << host_index_after[i] << std::endl;
            break;
        }
        used[host_index_after[i]] = true;
    }
    assert(permutation_valid);

    // Verify that grid values match the permutation
    bool values_match = true;
    for (int i = 0; i < N; i++) {
        if (host_grid_after[i] != host_grid[host_index_after[i]]) {
            values_match = false;
            if (verbose) {
                std::cout << "Mismatch at position " << i << ": got " << host_grid_after[i]
                    << " but expected " << host_grid[host_index_after[i]] << std::endl;
            }
            break;
        }
    }
    assert(values_match);

    if (permutation_valid && values_match) {
        std::cout << "Group-swap-degroup sequence test passed!" << std::endl;
    }

    // Cleanup
    delete[] host_grid;
    delete[] host_target_grid;
    delete[] host_index;
    delete[] host_grid_after;
    delete[] host_target_grid_after;
    delete[] host_index_after;
    delete[] host_group_perm;
    delete[] used;
    cudaFree(blockified_grid[0]);
    cudaFree(blockified_grid[1]);
    cudaFree(blockified_target_grid[0]);
    cudaFree(blockified_target_grid[1]);
    cudaFree(blockified_index[0]);
    cudaFree(blockified_index[1]);
    cudaFree(dist_to_target);
}

int main() {
    test_blockify();
    test_group(4, 4, 1, 1337, true);
    test_swap_within_group(4, 4, 1, true);
    test_group_swap_sequence(4, 4, 1, true);
    return 0;
}