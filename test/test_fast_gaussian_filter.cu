#include "filters.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>
#include <cassert>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

// Test fast_gaussian_filter on a random RGB image
void test_random_rgb() {
    int width = 256;
    int height = 256;
    int channels = 3;
    float sigma = 3.0f;
    int num_iterations = 6;
    FastGaussianFilter filter(height, width, channels, num_iterations);

    // Generate a random RGB image with values in range [0, 255]
    Mat random_img(height, width, CV_8UC3);
    randu(random_img, Scalar(0,0,0), Scalar(255,255,255));
    random_img.convertTo(random_img, CV_32FC3);

    // Allocate device memory
    size_t img_size = width * height * channels * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // Copy the random image to the device
    cudaMemcpy(d_input, random_img.ptr<float>(), img_size, cudaMemcpyHostToDevice);

    // Apply the fast gaussian filter
    filter.apply(d_input, d_output, sigma * 2.0);
    cudaDeviceSynchronize();

    // Retrieve the filtered image from device
    Mat filtered_img(height, width, CV_32FC3);
    cudaMemcpy(filtered_img.ptr<float>(), d_output, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Convert images for display in 8-bit
    Mat random_img_disp, filtered_img_disp;
    random_img.convertTo(random_img_disp, CV_8UC3);
    filtered_img.convertTo(filtered_img_disp, CV_8UC3);

    // Display the original and filtered images
    imshow("Random RGB Original", random_img_disp);
    imshow("Random RGB Filtered", filtered_img_disp);
    waitKey(0);
    destroyAllWindows();
}

// Test fast_gaussian_filter on a point distribution image
void test_point_distribution() {
    int width = 32;
    int height = 32;
    int channels = 3;
    float sigma = 3.0f;
    int num_iterations = 4;
    FastGaussianFilter filter(width, height, channels, 1);

    // Create a black image and add white points
    Mat point_img(height, width, CV_32FC3, Scalar(0, 0, 0));

    point_img.at<Vec3f>(height / 2, width / 2) = Vec3f(1.0f, 1.0f, 1.0f);

    // Allocate device memory
    size_t img_size = width * height * channels * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    // Copy the point distribution image to the device
    cudaMemcpy(d_input, point_img.ptr<float>(), img_size, cudaMemcpyHostToDevice);

    // Apply the fast gaussian filter
    filter.apply(d_input, d_output, sigma * 2.0);
    cudaDeviceSynchronize();

    // Retrieve the filtered image from device
    Mat filtered_img(height, width, CV_32FC3);
    cudaMemcpy(filtered_img.ptr<float>(), d_output, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Convert images for display in 8-bit
    Mat point_img_disp, filtered_img_disp;
    point_img.convertTo(point_img_disp, CV_8UC3, 255.0);
    filtered_img.convertTo(filtered_img_disp, CV_8UC3, 255.0);

    // Display the original and filtered images
    imshow("Point Distribution Original", point_img_disp);
    imshow("Point Distribution Filtered", filtered_img_disp);
    waitKey(0);
    destroyAllWindows();
}

int main() {
    // cout << "Testing fast_gaussian_filter on a random RGB image." << endl;
    // test_random_rgb();
    cout << "Testing fast_gaussian_filter on a point distribution image." << endl;
    test_point_distribution();
    return 0;
}