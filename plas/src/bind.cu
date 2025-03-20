#include "permuters.cuh"
#include "plas.cuh"
#include <random>
#include <string>
#include <torch/extension.h>

namespace plas {

    at::Tensor random_philox_permutation_cpu(int64_t n, int64_t num_rounds,
        at::Tensor& dummy) {
        // TODO: Implement the CPU version of the random philox permutation

        // Create a Mersenne Twister random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create a tensor with integers from 0 to n-1
        auto permutation = torch::arange(n, torch::kInt32);

        // Fisher-Yates shuffle algorithm
        for (int i = n - 1; i > 0; --i) {
            // Generate a random index between 0 and i (inclusive)
            std::uniform_int_distribution<int> dis(0, i);
            int j = dis(gen);

            // Swap elements at positions i and j
            auto temp = permutation[i].item<int>();
            permutation[i] = permutation[j];
            permutation[j] = temp;
        }

        return permutation;
    }

    at::Tensor random_philox_permutation_torch_cuda_wrapper(
        const int64_t n, const int64_t num_rounds, at::Tensor& dummy) {
        std::vector<int> keys = get_random_philox_keys(num_rounds);
        int* permutation = philox_permutation_cuda(n, num_rounds, keys);
        cudaDeviceSynchronize();
        std::function<void(void*)> deleter = [](void* ptr) { cudaFree(ptr); };
        return torch::from_blob(
            permutation, { n }, deleter,
            at::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
    }

    void sort_with_plas_torch_cuda_wrapper(
        const at::Tensor& grid, at::Tensor& grid_output, at::Tensor& index_output, at::Tensor& grid_target, const int64_t seed, const std::string& permuter_type,
        const int64_t filter_algo, const int64_t min_block_side, const int64_t min_filter_side_length,
        const double filter_decrease_factor, const double improvement_break,
        const int64_t min_group_configs, const int64_t max_group_configs,
        const bool verbose) {
        float* grid_target_ptr;
        if (grid_target.numel() == 0) {
            grid_target_ptr = nullptr;
        } else {
            grid_target_ptr = grid_target.data_ptr<float>();
        }
        RandomPermuter permuter;
        if (permuter_type == "lcg") {
            permuter = new LCGPermuter();
        } else {
            permuter = new PhiloxPermuter();
        }

        sort_with_plas(
            grid.data_ptr<float>(), grid_output.data_ptr<float>(), index_output.data_ptr<int>(), grid_target_ptr, grid.size(0), grid.size(1), grid.size(2), seed,
            permuter, filter_algo, min_block_side, min_filter_side_length, filter_decrease_factor,
            improvement_break, min_group_configs, max_group_configs, verbose);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(plas, m) {
        m.def("random_philox_permutation(int n, int num_rounds, Tensor dummy) -> "
            "Tensor");
        m.def("sort_with_plas(Tensor grid, \
                              Tensor grid_output, \
                              Tensor index_output, \
                              Tensor grid_target, \
                              int seed, \
                              str permuter_type, \
                              int filter_algo, \
                              int min_block_side, \
                              int min_filter_side_length, \
                              float filter_decrease_factor, \
                              float improvement_break, \
                              int min_group_configs, \
                              int max_group_configs, \
                              bool verbose) -> ()");
    }

    TORCH_LIBRARY_IMPL(plas, CPU, m) {
        m.impl("random_philox_permutation", &random_philox_permutation_cpu);
    }

    TORCH_LIBRARY_IMPL(plas, CUDA, m) {
        m.impl("random_philox_permutation",
            &random_philox_permutation_torch_cuda_wrapper);
        m.impl("sort_with_plas", &sort_with_plas_torch_cuda_wrapper);
    }
} // namespace plas
