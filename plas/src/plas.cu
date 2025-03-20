#include <assert.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h> 

#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <string>
#include "filters.cuh"
#include "plas.cuh"

int max_shared_mem_per_block;

std::vector<int> get_filter_side_lengths(int larger_grid_side,
    int min_filter_side_length,
    float decrease_factor) {
    std::vector<int> filter_side_lengths;
    float filter_side_length = larger_grid_side * decrease_factor;

    while (filter_side_length >= min_filter_side_length) {
        filter_side_lengths.push_back(int(filter_side_length));
        filter_side_length *= decrease_factor;
    }
    return filter_side_lengths;
}

__global__ void set_identity_permutation_kernel(int* tensor, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
        tensor[x] = x;
    }
}

void set_identity_permutation(int* tensor, int size) {
    int threads_per_block = 1024;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    set_identity_permutation_kernel << <num_blocks, threads_per_block >> > (tensor,
        size);
    cudaDeviceSynchronize();
}

__global__ void gather_kernel(const float* input, float* output,
    int* permutation, int H, int W, int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H * W) {
        for (int d = 0; d < C; d++) {
            output[index * C + d] = input[permutation[index] * C + d];
        }
    }
}

void gather(const float* input, float* output, int* permutation, int H, int W,
    int C) {
    int threads_per_block = 1024;
    int num_blocks = (H * W + threads_per_block - 1) / threads_per_block;
    gather_kernel << <num_blocks, threads_per_block >> > (input, output, permutation,
        H, W, C);
    cudaDeviceSynchronize();
}

__global__ void gather_kernel_int(const int* input, int* output,
    int* permutation, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H * W) {
        output[index] = input[permutation[index]];
    }
}

void gather(const int* input, int* output, int* permutation, int H, int W) {
    int threads_per_block = 1024;
    int num_blocks = (H * W + threads_per_block - 1) / threads_per_block;
    gather_kernel_int << <num_blocks, threads_per_block >> > (input, output,
        permutation, H, W);
    cudaDeviceSynchronize();
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__ void blockify_kernel(const float* in, float* out,
    const float* target_in, float* target_out,
    const int* index_in, int* index_out, int H, int W, int C,
    int block_len_x, int block_len_y, int tx,
    int ty, bool reverse) {
    int block_size = block_len_x * block_len_y;
    int num_blocks_x = (W - tx) / block_len_x;
    int num_blocks_y = (H - ty) / block_len_y;
    int num_blocks = num_blocks_x * num_blocks_y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= num_blocks * block_size)
        return;

    // Calculate read start position
    int x = thread_idx % (num_blocks_x * block_len_x);
    int y = thread_idx / (num_blocks_x * block_len_x);
    int read_start = y * (num_blocks_x * block_len_x) + x;
    read_start += ty * W;       // accomodate for ty
    read_start += (y + 1) * tx; // accomodate for the left margin caused by tx
    read_start += y * (W - (num_blocks_x * block_len_x + tx)); // accomodate for the right margin caused by tx

    // Calculate write start position
    int block_x = x / block_len_x;
    int block_y = y / block_len_y;
    int block_idx = block_y * num_blocks_x + block_x;
    int write_start = block_idx * block_len_x * block_len_y;
    write_start += (y - block_y * block_len_y) * block_len_x;
    write_start += (x - block_x * block_len_x);

    if (reverse) swap(write_start, read_start);

    for (int c = 0; c < C; c++) {
        out[write_start * C + c] = in[read_start * C + c];
        target_out[write_start * C + c] = target_in[read_start * C + c];
    }
    index_out[write_start] = index_in[read_start];
}

void blockify(const float* grid, float* blockified_grid,
    const float* target_grid, float* blockified_target_grid,
    const int* index, int* blockified_index, int H, int W, int C,
    int block_len_x, int block_len_y, int tx, int ty) {
    // Calculate number of blocks needed in output
    int num_blocks_x = (W - tx) / block_len_x;
    int num_blocks_y = (H - ty) / block_len_y;
    int total_blocks = num_blocks_x * num_blocks_y;
    int block_size = block_len_x * block_len_y;

    // Launch kernel with one thread per output position
    int threads_per_block = 1024;
    int num_blocks =
        (total_blocks * block_size + threads_per_block - 1) / threads_per_block;
    blockify_kernel << <num_blocks, threads_per_block >> > (
        grid, blockified_grid, target_grid, blockified_target_grid, index,
        blockified_index, H, W, C, block_len_x, block_len_y, tx, ty, false);
    cudaDeviceSynchronize();
}

void unblockify(float* grid, const float* blockified_grid, float* target_grid,
    const float* blockified_target_grid, int* index,
    const int* blockified_index, int H, int W, int C,
    int block_len_x, int block_len_y, int tx, int ty) {
    // Calculate number of blocks needed in output
    int num_blocks_x = (W - tx) / block_len_x;
    int num_blocks_y = (H - ty) / block_len_y;
    int total_blocks = num_blocks_x * num_blocks_y;
    int block_size = block_len_x * block_len_y;

    // Launch kernel with one thread per output position
    int threads_per_block = 1024;
    int num_blocks =
        (total_blocks * block_size + threads_per_block - 1) / threads_per_block;
    blockify_kernel << <num_blocks, threads_per_block >> > (
        blockified_grid, grid, blockified_target_grid, target_grid, 
        blockified_index, index, H, W, C, block_len_x, block_len_y, tx, ty, true);
    cudaDeviceSynchronize();
}

__global__ void group_kernel(float* blockified_grid_read,
    float* blockified_grid_write,
    float* blockified_target_grid_read,
    float* blockified_target_grid_write,
    int* blockified_index_read,
    int* blockified_index_write, int* permutation,
    int block_size, int num_blocks, int C, int turn) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_blocks * block_size)
        return;

    int block_idx = thread_idx / block_size;
    int block_offset = thread_idx % block_size;
    int read_start = block_idx * block_size + permutation[block_offset];
    int write_start = block_idx * block_size + block_offset;
    for (int c = 0; c < C; c++) {
        blockified_grid_write[write_start * C + c] =
            blockified_grid_read[read_start * C + c];
        blockified_target_grid_write[write_start * C + c] =
            blockified_target_grid_read[read_start * C + c];
    }
    blockified_index_write[write_start] =
        blockified_index_read[read_start];
}

void group(float* blockified_grid[2], float* blockified_target_grid[2],
    int* blockified_index[2], int* permutation, int block_size,
    int num_blocks, int C, int turn) {
    int threads_per_block = 1024;
    int num_cuda_blocks =
        (num_blocks * block_size + threads_per_block - 1) / threads_per_block;
    group_kernel << <num_cuda_blocks, threads_per_block >> > (
        blockified_grid[turn], blockified_grid[1 - turn],
        blockified_target_grid[turn], blockified_target_grid[1 - turn],
        blockified_index[turn], blockified_index[1 - turn], permutation,
        block_size, num_blocks, C, turn);
    cudaDeviceSynchronize();
}

inline __device__ int get(int perm, int i) {
    perm >>= 8 * i;
    return perm & int(255);
}

inline __device__ void set(int& perm, int i, int value) {
    perm &= ~(int(255) << (8 * i));
    perm |= value << (8 * i);
}

inline __device__ void swap(int& perm, int i, int j) {
    int i_val = get(perm, i);
    int j_val = get(perm, j);
    set(perm, i, j_val);
    set(perm, j, i_val);
}

inline __device__ float compute_dist(float* group_data, float* target_group_data, int perm, int C) {
    float dist = 0;
    for (int i = 0; i < 4; i++) {
        int j = get(perm, i);
        for (int c = 0; c < C; c++) {
            float delta = group_data[j * C + c] - target_group_data[i * C + c];
            dist += delta * delta;
        }
    }
    return dist;
}

__global__ void
swap_within_group_kernel(float* blockified_grid, float* blockified_target_grid,
    float* dist_to_target, int* blockified_index, int C, int num_groups) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups) return;

    // copy all data used by the block into shared memory
    extern __shared__ float shared_data[];
    float* grid_shared = shared_data;
    float* target_grid_shared = shared_data + blockDim.x * 4 * C;
    for (int i = 0; i < 4; i++) {
        for (int c = 0; c < C; c++) {
            int local_idx = threadIdx.x * 4 * C + i * C + c;
            int global_idx = group_idx * 4 * C + i * C + c;
            grid_shared[local_idx] = blockified_grid[global_idx];
            target_grid_shared[local_idx] = blockified_target_grid[global_idx];
        }
    }

    // store all permutations and their associated distances in registers
    // in order to avoid branch divergence and pipeline stalls which would happen
    // if we maintained the best permutation and distance
    int permutations[24];
    float distances[24];

    // start with the identity permutation
    int perm_counter = 0;
    int perm = 0;
    for (int i = 0; i < 4; i++) set(perm, i, i);
    distances[perm_counter] = compute_dist(grid_shared + threadIdx.x * 4 * C, target_grid_shared + threadIdx.x * 4 * C, perm, C);
    permutations[perm_counter++] = perm;

    // iterate over all the other permutations with only one swap per iteration
    // by using Heap's algorithm (https://en.wikipedia.org/wiki/Heap%27s_algorithm)
    int stack_state[4] = { 0, 0, 0, 0 };
    int i = 1;
    while (i < 4) {
        if (stack_state[i] < i) {
            if (i % 2 == 0) {
                swap(perm, 0, i);
            }
            else {
                swap(perm, stack_state[i], i);
            }

            distances[perm_counter] = compute_dist(grid_shared + threadIdx.x * 4 * C, target_grid_shared + threadIdx.x * 4 * C, perm, C);
            permutations[perm_counter++] = perm;

            stack_state[i]++;
            i = 1;
        }
        else {
            stack_state[i] = 0;
            i++;
        }
    }

    // find the permutation with the smallest distance
    float best_dist = distances[0];
    int best_perm = permutations[0];
    for (int i = 1; i < 24; i++) {
        if (distances[i] < best_dist) {
            best_perm = permutations[i];
            best_dist = distances[i];
        }
    }

    // permute blockified grid according to the best permutation
    for (int i = 0; i < 4; i++) {
        for (int c = 0; c < C; c++) {
            int p_i = get(best_perm, i);
            blockified_grid[group_idx * 4 * C + i * C + c] = grid_shared[threadIdx.x * 4 * C + get(best_perm, i) * C + c];
        }
    }

    // also permute blockified index, use grid_shared as a temporary buffer
    int* temp = (int*)(grid_shared + threadIdx.x * 4 * C);
    for (int i = 0; i < 4; i++) {
        temp[i] = blockified_index[group_idx * 4 + i];
    }
    for (int i = 0; i < 4; i++) {
        blockified_index[group_idx * 4 + i] = temp[get(best_perm, i)];
    }

    // write the best distance to global memory
    dist_to_target[group_idx] = best_dist;
}

void swap_within_group(float* blockified_grid, float* blockified_target_grid,
    float* dist_to_target, int* blockified_index, int C, int num_groups) {
    int shared_mem_per_thread = 2 * 4 * C * sizeof(float);
    int threads_per_block = std::min<int>(1024, max_shared_mem_per_block / shared_mem_per_thread);
    int shared_mem_size = threads_per_block * shared_mem_per_thread;
    int num_cuda_blocks =
        (num_groups + threads_per_block - 1) / threads_per_block;
    swap_within_group_kernel << <num_cuda_blocks, threads_per_block, shared_mem_size >> > (
        blockified_grid, blockified_target_grid, dist_to_target, blockified_index,
        C, num_groups);
    cudaDeviceSynchronize();
}

float average(float* d_begin, float* d_end, bool use_thrust = true) {
    if (use_thrust) {
        return thrust::reduce(thrust::device, d_begin, d_end, 0.0, thrust::plus<float>()) / (d_end - d_begin);
    } else {
        // naive host fallback
        float* h_dist_to_target = new float[d_end - d_begin];
        cudaMemcpy(h_dist_to_target, d_begin, (d_end - d_begin) * sizeof(float), cudaMemcpyDeviceToHost);
        float current_dist_to_target = 0;
        for (int i = 0; i < d_end - d_begin; i++) {
            current_dist_to_target += h_dist_to_target[i];
        }
        current_dist_to_target /= (d_end - d_begin);
        return current_dist_to_target;
    }
}

bool are_grid_and_permutation_consistent(const float* input_grid, float* grid, int* index, int H, int W, int C) {
    float* host_input_grid = new float[H * W * C];
    float* host_grid = new float[H * W * C];
    int* host_index = new int[H * W];
    cudaMemcpy(host_input_grid, input_grid, H * W * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_grid, grid, H * W * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_index, index, H * W * sizeof(int), cudaMemcpyDeviceToHost);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                int p = host_index[h * W + w];
                if (abs(host_grid[(h * W + w) * C + c] - host_input_grid[p * C + c]) > 1e-4) return false;
            }
        }
    }
    return true;
}

void query_maximum_shared_memory_per_block() {
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    max_shared_mem_per_block = prop.sharedMemPerBlock;
}


bool is_permutation_device(int* permutation, int N) {
    int* host_permutation = new int[N];
    cudaMemcpy(host_permutation, permutation, N * sizeof(int), cudaMemcpyDeviceToHost);
    bool* seen = new bool[N]();
    for (int i = 0; i < N; i++) {
        if (host_permutation[i] < 0 || host_permutation[i] >= N) {
            return false;
        }
        if (seen[host_permutation[i]]) {
            return false;
        }
        seen[host_permutation[i]] = true;
    }
    delete[] seen;
    delete[] host_permutation;
    return true;
}

bool print_permutation_device(int* permutation, int N) {
    int* host_permutation = new int[N];
    cudaMemcpy(host_permutation, permutation, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        std::cerr << host_permutation[i] << " ";
    }
    std::cerr << std::endl;
    delete[] host_permutation;
    return true;
}

void sort_with_plas(const float* grid_input, // of size H x W x C, memory layout:
    // grid_input[h * W * C + w * C + c]
    float* grid_output, // same size as grid_input
    int* index_output, // same layout as grid_input but without channels,
    float* grid_target,
    int H, int W, int C, int64_t seed, RandomPermuter* permuter, int filter_algo,
    int min_block_side, int min_filter_side_length, float filter_decrease_factor,
    float improvement_break, int min_group_configs,
    int max_group_configs, bool verbose) {
    Filter* filter;
    if (filter_algo == 0) {
        filter = new FastGaussianFilter(W, H, C);
    } else {
        filter = new GaussianFilterViaCVCUDA(W, H, C);
    }

    permuter->set_seed(seed);
    query_maximum_shared_memory_per_block();

    float* blockified_grid[2];
    blockified_grid[0] = get_2d_tensor_device_memory<float>(H * W, C);
    blockified_grid[1] = get_2d_tensor_device_memory<float>(H * W, C);
    // Permuting data across different CUDA blocks prohibits in-place
    // permuting. Therefore we need to use two buffers to store the data. At index
    // $turn is the data to be processed next.
    bool turn = 0;

    // target_grid is the idealized target grid if no fixed target is provided
    if (grid_target == nullptr) {
        grid_target = get_3d_tensor_device_memory<float>(H, W, C);
    }
    float* blockified_target_grid[2];
    blockified_target_grid[0] = get_2d_tensor_device_memory<float>(H * W, C);
    blockified_target_grid[1] = get_2d_tensor_device_memory<float>(H * W, C);
    float* dist_to_target =
        get_1d_tensor_device_memory<float>(H * W); // also blockified

    // index maintains how pixels are reordered in reference to input_grid
    int* blockified_index[2];
    blockified_index[0] = get_1d_tensor_device_memory<int>(H * W);
    blockified_index[1] = get_1d_tensor_device_memory<int>(H * W);
    set_identity_permutation(index_output, H * W);

    // start with an initial random reordering to not end up in a bad local
    // minimum
    int* permutation = permuter->get_new_permutation(H * W);
    gather(grid_input, grid_output, permutation, H, W, C);
    permuter->reset_fusing_parameters();

    cudaMemcpy(index_output, permutation, H * W * sizeof(int), cudaMemcpyDeviceToDevice);

    std::uniform_int_distribution<int> alignment_sampler(0, 1);
    std::mt19937 rng(seed);

    std::vector<int> filter_side_lengths = get_filter_side_lengths(
        min(H, W), min_filter_side_length, filter_decrease_factor);
    for (int filter_side_length : filter_side_lengths) {
        // Blur the grid to obtain an idealized target 
        int filter_size = filter_side_length;
        if (filter_size % 2 == 0)
            ++filter_size;
        if (grid_target != nullptr) {
            filter->apply(grid_output, grid_target, filter_size);
        }

        int block_len_x = filter_size - 1;
        int block_len_y = filter_size - 1;
        int block_size = block_len_x * block_len_y;
        int num_blocks = (W / block_len_x) * (H / block_len_y);
        // Draw a random alignment of the blocks in the grid.
        bool is_left_aligned_x = alignment_sampler(rng);
        bool is_left_aligned_y = alignment_sampler(rng);
        int tx = is_left_aligned_x ? 0 : (W % block_len_x);
        int ty = is_left_aligned_y ? 0 : (H % block_len_y);

        // Write the blocks sequentially to the blockified grid for more efficient
        // memory access patterns.
        blockify(grid_output, blockified_grid[turn], grid_target,
            blockified_target_grid[turn], index_output, blockified_index[turn], H, W,
            C, block_len_x, block_len_y, tx, ty);

        // Try a few random group configurations until the improvement is too small,
        // or we have tried enough configurations. But try at least
        // min_group_configs.
        float previous_dist_to_target = std::numeric_limits<float>::max();
        float improvement = 0.0;
        for (int j = 0; j < max_group_configs &&
            (improvement > improvement_break || j < min_group_configs);
            j++) {
            // Group all pixels inside a block into groups of 4 pixels based on a
            // random permutation.
            int* permutation = permuter->get_new_permutation(block_size);

            group(blockified_grid, blockified_target_grid, blockified_index,
                permutation, block_size, num_blocks, C, turn);
            turn = 1 - turn;

            // For each group, find the best permutation out of the 24 possible
            // ones judged by the idealized target grid.
            swap_within_group(blockified_grid[turn], blockified_target_grid[turn],
                dist_to_target, blockified_index[turn], C, num_blocks * block_size / 4);

            // Imediately apply the inverse permutation if the permuter does not
            // support inverse fusion.
            if (!permuter->supports_inverse_fusion()) {
                int* inverse_permutation =
                    permuter->get_inverse_permutation(block_size);
                group(blockified_grid, blockified_target_grid, blockified_index,

                    inverse_permutation, block_size, num_blocks, C, turn);
                turn = 1 - turn;
            }

            float current_dist_to_target = average(dist_to_target, dist_to_target + num_blocks * block_size / 4) / 4;

            improvement = previous_dist_to_target - current_dist_to_target;
            previous_dist_to_target = current_dist_to_target;
        }

        // Apply the inverse permutation if the permuter supports inverse fusion.
        if (permuter->supports_inverse_fusion()) {
            int* inverse_permutation = permuter->get_inverse_permutation(block_size);
            group(blockified_grid, blockified_target_grid, blockified_index,
                inverse_permutation, block_size, num_blocks, C, turn);
            turn = 1 - turn;
        }


        // Write the blocks back to the grid.
        unblockify(grid_output, blockified_grid[turn], grid_target,
            blockified_target_grid[turn], index_output, blockified_index[turn], H,
            W, C, block_len_x, block_len_y, tx, ty);

    }

    // Free all intermediate tensors.
    if (grid_target != nullptr) {
        cudaFree(grid_target);
    }
    cudaFree(blockified_grid[0]);
    cudaFree(blockified_grid[1]);
    cudaFree(blockified_target_grid[0]);
    cudaFree(blockified_target_grid[1]);
    cudaFree(blockified_index[0]);
    cudaFree(blockified_index[1]);
}

std::pair<float*, int*> sort_with_plas(
    const float* grid_input,
    int H, int W, int C, int64_t seed, float* grid_target, RandomPermuter* permuter, int filter_algo, int min_block_side, int min_filter_side_length,
    float filter_decrease_factor, float improvement_break,
    int min_group_configs, int max_group_configs,
    bool verbose) {
    float* grid_output = get_3d_tensor_device_memory<float>(H, W, C);
    int* index_output = get_2d_tensor_device_memory<int>(H, W);
    sort_with_plas(grid_input, grid_output, index_output, grid_target, H, W, C, seed, permuter, filter_algo, min_block_side, min_filter_side_length, filter_decrease_factor, improvement_break, min_group_configs, max_group_configs, verbose);
    return std::make_pair(grid_output, index_output);
}