#pragma once
#include <cuda_runtime.h>
#include <vector>

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
