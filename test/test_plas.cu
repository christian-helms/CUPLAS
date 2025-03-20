#include "plas.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>
#include <cassert>
#include <iostream>
#include <random>

template <typename T>
T* to_host(T* d, int size) {
    T* h = new T[size];
    cudaMemcpy(h, d, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return h;
}

template <typename T>
T* to_device(T* h, int size) {
    T* d;
    cudaMalloc(&d, size * sizeof(T));
    cudaMemcpy(d, h, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return d;
}

bool is_permutation(int* candidate_h, int size) {
    bool seen[size] = {false};
    for (int i = 0; i < size; i++) {
        if (candidate_h[i] < 0 || candidate_h[i] >= size || seen[candidate_h[i]]) {
            return false;
        }
        seen[candidate_h[i]] = true;
    }
    return true;
}

bool grid_is_sorted_according_to_index(float* input_grid_h, float* sorted_grid_h, int* index_h, int H, int W, int C) {
    for (int i = 0; i < H * W; i++) {
        for (int c = 0; c < C; c++) {
            if (std::abs(sorted_grid_h[i * C + c] - input_grid_h[index_h[i] * C + c]) > 1e-9) {
                return false;
            }
        }
    }
    return true;
}

void save_image(float* grid_h, int H, int W, int C, std::string path) {
    cv::Mat image(H, W, CV_32FC(C), grid_h);
    image *= 255.0f;
    cv::Mat image_8u;
    image.convertTo(image_8u, CV_8U);
    try {
        std::string output_path = path;
        bool write_success = cv::imwrite(output_path, image_8u);
        if (!write_success) {
            std::cerr << "Failed to write image to: " << output_path << std::endl;
        } else {
            std::cout << "Successfully wrote image to: " << output_path << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error writing image: " << e.what() << std::endl;
    }
}

float* get_comparison_image(float* input_grid_h, float* sorted_grid_h, int H, int W, int C) {
    float* comparison_image = new float[H * W * C * 2];
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            for (int channel = 0; channel < C; channel++) {
                comparison_image[row * 2 * W * C + col * C + channel] = input_grid_h[row * W * C + col * C + channel];
                comparison_image[row * 2 * W * C + col * C + channel + W * C] = sorted_grid_h[row * W * C + col * C + channel];
            }
        }
    }
    return comparison_image;
}

void test_on_random_image(int H, int W, int C, int64_t seed, bool save = false) {
    int size = H * W * C;
    float* input_grid_h = new float[size];
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, 1);
    for (int i = 0; i < size; i++) {
        input_grid_h[i] = dist(rng);
    }
    float* input_grid_d = to_device(input_grid_h, size);

    auto [sorted_grid_d, sorted_index_d] = sort_with_plas(input_grid_d, H, W, C, seed);
 
    float* sorted_grid_h = to_host(sorted_grid_d, size);
    int* sorted_index_h = to_host(sorted_index_d, H * W);

    if (save) {
        float* comparison_image = get_comparison_image(input_grid_h, sorted_grid_h, H, W, C);
        save_image(comparison_image, H, 2 * W, C, "/home/chris/Dev/PLAS-Gitlab/test/sorted_comparison.png");
        delete[] comparison_image;
    }

    assert(is_permutation(sorted_index_h, H * W));

    assert(grid_is_sorted_according_to_index(input_grid_h, sorted_grid_h, sorted_index_h, H, W, C));

    delete[] input_grid_h;
    delete[] sorted_grid_h;
    delete[] sorted_index_h;
    cudaFree(input_grid_d);
    cudaFree(sorted_grid_d);
    cudaFree(sorted_index_d);
}

void test_grayscale_small() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(4, 20);
    for (int64_t seed = 0; seed < 10; seed++) {
        int H = dist(rng);
        int W = dist(rng);
        test_on_random_image(H, W, 1, seed);
    }
}

void test_grayscale_medium() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(100, 200);
    for (int64_t seed = 0; seed < 2; seed++) {
        int H = dist(rng);
        int W = dist(rng);
        test_on_random_image(H, W, 1, seed);
    }
}

void test_grayscale_large() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(500, 1000);
    for (int64_t seed = 0; seed < 2; seed++) {
        int H = dist(rng);
        int W = dist(rng);
        test_on_random_image(H, W, 1, seed);
    }
}

void test_rgb_small() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(4, 4);
    for (int64_t seed = 0; seed < 10; seed++) {
        int H = dist(rng);
        int W = dist(rng);
        test_on_random_image(H, W, 3, seed);
    }
}

void test_rgb_medium() {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(100, 200);
    for (int64_t seed = 0; seed < 2; seed++) {
        int H = dist(rng);
        int W = dist(rng);
        test_on_random_image(H, W, 3, seed);
    }
}

int main() {
    // test_grayscale_small();
    // test_grayscale_medium();
    // test_grayscale_large();
    // test_rgb_small();
    // test_rgb_medium();

    test_on_random_image(4096, 4096, 3, 1337, false);

    std::cout << "All tests passed!" << std::endl;
    return 0;
}