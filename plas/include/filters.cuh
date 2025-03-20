#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <nvcv/Tensor.hpp>
#include <cvcuda/Operator.h>
#include <cvcuda/OpGaussian.h>

class Filter {
public:
    virtual void apply(const float* d_input, float* d_output, int filter_size) = 0;
};

class FastGaussianFilter : public Filter {
public:
    FastGaussianFilter(int width, int height, int channels, int num_iterations = 3);
    ~FastGaussianFilter();

    void apply(const float* d_input, float* d_output, int filter_size);

private:
    int W;
    int H;
    int C;
    float sigma_;
    int num_iterations_;

    float* C_major_buffer_in;
    float* C_major_buffer_out;

    size_t max_shared_memory_per_block;

    void to_C_major(const float* d_input);
    void to_C_minor(float* d_output);
    void row_wise_iterative_box_filter(const float* d_input, float* d_output, int filter_size);
    void transpose_single_channel_grid(const float* d_input, float* d_output);
};

class GaussianFilterViaCVCUDA : public Filter {
public:
    GaussianFilterViaCVCUDA( int width, int height, int channels);
    ~GaussianFilterViaCVCUDA();

    // Applies the Gaussian filter with the given kernel size. 
    // d_input and d_output are device pointers with channel-major layout HWC.
    void apply(const float* d_input, float* d_output, int filter_size);

private:
    int width_;
    int height_;
    int channels_;
    
    cudaStream_t stream_;
    NVCVOperatorHandle op_;

    // Preallocated temporary buffers on GPU:
    // d_gather_buffer_ will hold gathered group data; d_filter_buffer_ will store the filtered result
    float* d_gather_buffer_;
    float* d_filter_buffer_;

    // Structure to describe a group of channels
    struct Group {
        int start_channel; // starting channel index in the interleaved input
        int group_size;    // number of channels in this group
    };

    // The groups determined by decomposing "channels_" into sizes that NPPI supports (1, 3, or 4).
    std::vector<Group> groups_;

    // Computes the grouping of channels (each group size is one of {4,3,1}) that minimizes the number of groups.
    std::vector<int> compute_groups(int C);

    // Initializes "groups_" using compute_groups
    void init_groups();
};
