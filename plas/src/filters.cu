#include <cassert>
#include <iostream>
#include <vector>
#include "filters.cuh"

///////////////////////////////////////////////////////////////
// Fast Gaussian Filter implementation (apply function and helpers)
// CURRENTLY NOT OPERATIONAL (in development)
///////////////////////////////////////////////////////////////

int
maximum_shared_memory_per_block()
{
  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  return prop.sharedMemPerBlock;
}

int
box_size_for_gaussian_filter(float sigma, int n)
{
  float variance = sigma * sigma;
  int k = round(sqrt((variance * 12 + n) / n));
  if (k % 2 == 0) {
    k += 1;
  }
  return k;
}

__global__ void
transpose_grid(const float* input,
               float* output,
               int w,
               int h,
               int tile_dim,
               int rows_per_block)
{
  extern __shared__ float tile[];

  int x = blockIdx.x * tile_dim + threadIdx.x;
  int y = blockIdx.y * tile_dim + threadIdx.y;

  // copy tile to shared memory
  // TODO: add controlled padding to always avoid bank conflicts
  for (int j = 0; j < tile_dim; j += rows_per_block) {
    if (x < w && y + j < h) {
      tile[((threadIdx.y + j) * tile_dim + threadIdx.x)] =
        input[((y + j) * w + x)];
    }
  }

  __syncthreads();

  x = blockIdx.y * tile_dim + threadIdx.x; // transpose block offset
  y = blockIdx.x * tile_dim + threadIdx.y;

  for (int j = 0; j < tile_dim; j += rows_per_block) {
    if (x < w && y + j < h) {
      output[((y + j) * w + x)] =
        tile[(threadIdx.x * tile_dim + threadIdx.y + j)];
    }
  }
}

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define MAX_BLOCK_SIZE 1024

// Define this to more rigorously avoid bank conflicts, even at the lower (root)
// levels of the tree #define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index)                                            \
  ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

// #ifdef CHECK_BANK_CONFLICTS
// #define TEMP(index) CUT_BANK_CHECKER(temp, index)
// #else
// #define TEMP(index) temp[index]
// #endif

__device__ void
row_wise_inclusive_scan_in_place(float* temp)
{
  int W = 2 * blockDim.x;

  int ai = threadIdx.x;
  int bi = threadIdx.x + (W / 2);

  // compute spacing to avoid bank conflicts
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  float inputA = temp[ai + bankOffsetA];
  float inputB = temp[bi + bankOffsetB];

  int offset = 1;

  // build the sum in place up the tree
  for (int d = W / 2; d > 0; d >>= 1) {
    __syncthreads();

    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }

    offset *= 2;
  }

  // scan back down the tree
  __syncthreads();
  // clear the last element of the block and store the sum in the sums array
  if (threadIdx.x == 0) {
    temp[W - 1 + CONFLICT_FREE_OFFSET(W - 1)] = 0;
  }

  // traverse down the tree building the scan in place
  __syncthreads();
  for (int d = 1; d < W; d *= 2) {
    offset /= 2;

    __syncthreads();

    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  // For inclusive scan, add the original input values to the exclusive scan
  __syncthreads();
  temp[ai + bankOffsetA] += inputA;
  temp[bi + bankOffsetB] += inputB;
  __syncthreads();
}

__device__ float
calc_box_filter_element(float* temp_in, int idx, int W, int filter_size)
{
  // Half filter size for convenience
  int half_size = filter_size / 2;

  // Calculate true window boundaries (can go outside array bounds)
  int true_left = idx - half_size;
  int true_right = idx + half_size;

  // Calculate actual array indices to access (clamped to valid range)
  int left = max(0, true_left);
  int right = min(W - 1, true_right);

  float sum;

  // Calculate the sum within the array bounds
  if (left == 0) {
    sum = temp_in[right + CONFLICT_FREE_OFFSET(right)];
  } else {
    sum = temp_in[right + CONFLICT_FREE_OFFSET(right)] -
          temp_in[left - 1 + CONFLICT_FREE_OFFSET(left - 1)];
  }

  // Add contribution from replicated left border pixels
  if (true_left < 0) {
    // Add left border value (temp_in[0]) repeated (left - true_left) times
    float border_value = temp_in[0 + CONFLICT_FREE_OFFSET(0)];
    sum += border_value * (-true_left);
  }

  // Add contribution from replicated right border pixels
  if (true_right >= W) {
    // Add right border value (temp_in[W-1]) repeated (true_right - right) times
    float border_value;
    border_value = temp_in[W - 1 + CONFLICT_FREE_OFFSET(W - 1)] -
                   temp_in[W - 2 + CONFLICT_FREE_OFFSET(W - 2)];
    sum += border_value * (true_right - right);
  }

  // Window size is always filter_size regardless of boundary conditions
  return sum / filter_size;
}

__device__ void
row_wise_box_filter_replicate_border(float* temp_in,
                                     float* temp_out,
                                     int filter_size)
{
  int W = 2 * blockDim.x;
  int ai = threadIdx.x;
  int bi = threadIdx.x + (W / 2);

  // compute spacing to avoid bank conflicts
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Process element ai and bi using the helper function
  temp_out[ai + bankOffsetA] =
    calc_box_filter_element(temp_in, ai, W, filter_size);
  temp_out[bi + bankOffsetB] =
    calc_box_filter_element(temp_in, bi, W, filter_size);

  __syncthreads();
}

__global__ void
row_wise_iterative_box_filter_kernel(const float* g_input,
                                     float* g_output,
                                     int filter_size,
                                     int num_iterations)
{
  int W = 2 * blockDim.x;

  // Dynamically allocated shared memory - we'll partition it manually
  extern __shared__ float shared_block_mem[];

  // Manually partition the shared memory
  // First half of shared memory is temp[0], second half is temp[1]
  float* temp0 = shared_block_mem;
  float* temp1 = shared_block_mem + W + CONFLICT_FREE_OFFSET(W);
  float* temp[2] = { temp0, temp1 };
  int turn = 0;

  int ai = threadIdx.x;
  int bi = threadIdx.x + (W / 2);

  // compute spacing to avoid bank conflicts
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Cache the row in shared memory
  float inputA = g_input[ai + blockIdx.x * W];
  float inputB = g_input[bi + blockIdx.x * W];

  // Store in the first buffer
  temp[turn][ai + bankOffsetA] = inputA;
  temp[turn][bi + bankOffsetB] = inputB;

  // Process iterations with buffer swapping
  for (int i = 0; i < num_iterations; i++) {
    row_wise_inclusive_scan_in_place(temp[turn]);
    row_wise_box_filter_replicate_border(
      temp[turn], temp[1 - turn], filter_size);
    turn = 1 - turn;
  }

  // Write output from the current buffer
  g_output[ai + blockIdx.x * W] = temp[turn][ai + bankOffsetA];
  g_output[bi + blockIdx.x * W] = temp[turn][bi + bankOffsetB];
}

void
FastGaussianFilter::row_wise_iterative_box_filter(const float* d_input,
                                                  float* d_output,
                                                  int filter_size)
{
  int block_dim = W / 2;
  int grid_dim = H;
  int shared_mem_per_block = 2 * (W + CONFLICT_FREE_OFFSET(W)) * sizeof(float);

  row_wise_iterative_box_filter_kernel<<<grid_dim,
                                         block_dim,
                                         shared_mem_per_block>>>(
    d_input, d_output, filter_size, num_iterations_);
  cudaDeviceSynchronize();
}

// CUDA kernel: gather channels from interleaved input into a contiguous group
// buffer
__global__ void
gather_channels_kernel(const float* d_input,
                       float* d_gather,
                       int num_pixels,
                       int full_channels,
                       int start_channel,
                       int group_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_pixels) {
    int in_offset = idx * full_channels + start_channel;
    int out_offset = idx * group_size;
    for (int i = 0; i < group_size; i++) {
      d_gather[out_offset + i] = d_input[in_offset + i];
    }
  }
}

// CUDA kernel: scatter channels from contiguous group buffer back to
// interleaved output
__global__ void
scatter_channels_kernel(const float* d_group,
                        float* d_output,
                        int num_pixels,
                        int full_channels,
                        int start_channel,
                        int group_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_pixels) {
    int out_offset = idx * full_channels + start_channel;
    int in_offset = idx * group_size;
    for (int i = 0; i < group_size; i++) {
      d_output[out_offset + i] = d_group[in_offset + i];
    }
  }
}

void
FastGaussianFilter::to_C_major(const float* d_input)
{
  int num_pixels = W * H;
  int threads = 256;
  for (int c = 0; c < C; c++) {
    int blocks = (num_pixels + threads - 1) / threads;
    gather_channels_kernel<<<blocks, threads>>>(
      d_input, C_major_buffer_in + c * W * H, num_pixels, C, c, 1);
  }
  cudaDeviceSynchronize();
}

void
FastGaussianFilter::to_C_minor(float* d_output)
{
  int num_pixels = W * H;
  int threads = 256;
  for (int c = 0; c < C; c++) {
    int blocks = (num_pixels + threads - 1) / threads;
    scatter_channels_kernel<<<blocks, threads>>>(
      C_major_buffer_in + c * W * H, d_output, num_pixels, C, c, 1);
  }
  cudaDeviceSynchronize();
}

void
FastGaussianFilter::transpose_single_channel_grid(const float* d_input,
                                                  float* d_output)
{
  int max_shared_mem_per_block = maximum_shared_memory_per_block();
  int tile_dim = int(sqrt(max_shared_mem_per_block / sizeof(float)));
  if (tile_dim >= 64)
    tile_dim = 64;
  else {
    assert(tile_dim >= 32);
    tile_dim = 32;
  }
  int shared_mem_per_block = tile_dim * tile_dim * sizeof(float);
  int rows_per_block = tile_dim / 4;
  dim3 num_blocks((W + tile_dim - 1) / tile_dim, (H + tile_dim - 1) / tile_dim);
  dim3 threads_per_block(tile_dim, rows_per_block);
  transpose_grid<<<num_blocks, threads_per_block, shared_mem_per_block>>>(
    d_input, d_output, W, H, tile_dim, rows_per_block);
  cudaDeviceSynchronize();
}

// void
// show_image(const float* d_img, int width, int height, int channels = 1)
// {
//   size_t img_size = width * height * channels;
//   float* h_img = new float[img_size];
//   cudaMemcpy(h_img, d_img, img_size * sizeof(float), cudaMemcpyDeviceToHost);

//   cv::Mat img_mat;
  
//   if (channels == 1) {
//     // Single-channel grayscale image
//     img_mat = cv::Mat(height, width, CV_32FC1, h_img);
    
//     // Convert to 8-bit for display
//     cv::Mat display_img;
//     img_mat.convertTo(display_img, CV_8U, 255.0); // Scale to full 8-bit range
    
//     // Display the grayscale image
//     cv::imshow("Raw Image Data (Grayscale)", display_img);
//   } 
//   else if (channels == 3) {
//     // 3-channel color image
//     img_mat = cv::Mat(height, width, CV_32FC3, h_img);
    
//     // Convert to 8-bit for display
//     cv::Mat display_img;
//     img_mat.convertTo(display_img, CV_8UC3, 255.0); // Scale to full 8-bit range
    
//     // OpenCV uses BGR order, so if your data is in RGB order, convert it
//     // Uncomment if your channel order is RGB instead of BGR
//     // cv::cvtColor(display_img, display_img, cv::COLOR_RGB2BGR);
    
//     // Display the color image
//     cv::imshow("Raw Image Data (Color)", display_img);
//   }
//   else {
//     std::cerr << "Unsupported number of channels: " << channels << std::endl;
//     delete[] h_img;
//     return;
//   }
  
//   cv::waitKey(0);
//   cv::destroyAllWindows();
  
//   // Clean up memory
//   delete[] h_img;
// }


void
FastGaussianFilter::apply(const float* input, float* output, int filter_size)
{
  float sigma = filter_size / 2.0f;
  int box_size = box_size_for_gaussian_filter(sigma, num_iterations_);

  to_C_major(input);

  for (int c = 0; c < C; c++) {
    float* buffer_in = C_major_buffer_in + c * W * H;
    float* buffer_out = C_major_buffer_out + c * W * H;

    row_wise_iterative_box_filter(buffer_in, buffer_out, box_size);
    // show_image(buffer_out, W, H, 1);
    transpose_single_channel_grid(buffer_out, buffer_in);
    // show_image(buffer_in, W, H, 1);
    row_wise_iterative_box_filter(buffer_in, buffer_out, box_size);
    // show_image(buffer_out, W, H, 1);
    transpose_single_channel_grid(buffer_out, buffer_in);
    // show_image(buffer_in, W, H, 1);
  }

  to_C_minor(output);
}

FastGaussianFilter::FastGaussianFilter(int width,
                                       int height,
                                       int channels,
                                       int num_iterations)
  : W{ width }
  , H{ height }
  , C{ channels }
  , num_iterations_{ num_iterations }
{
  cudaMalloc(&C_major_buffer_in, W * H * C * sizeof(float));
  cudaMalloc(&C_major_buffer_out, W * H * C * sizeof(float));
  max_shared_memory_per_block = maximum_shared_memory_per_block();
}

FastGaussianFilter::~FastGaussianFilter()
{
  cudaFree(C_major_buffer_in);
  cudaFree(C_major_buffer_out);
}
