#include "cudaops.h"

namespace Inferno {


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cross_entropy_loss_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    __global__ void cross_entropy_loss_kernel(const LT* logits,const int* targets,LT* out,size_t rows,size_t vocab_size) {
        size_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + (r * vocab_size);
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        LT max_logit = row_ptr[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row_ptr[v] > max_logit) {
                max_logit = row_ptr[v];
            }
        }

        LT sum_exp = static_cast<LT>(0);
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += exp(row_ptr[v] - max_logit);
        }

        LT log_sum_exp = log(sum_exp);
        LT target_logit = row_ptr[target_id];
        LT row_loss = -(target_logit - max_logit - log_sum_exp);

        atomicAdd(out, row_loss / static_cast<LT>(rows));
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    void cuda_cross_entropy_loss(const LT* logits, const int* targets, LT* out, size_t rows, size_t vocab_size) {

        cudaMemset(out, 0, sizeof(LT));

        const int threads = 256;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);

        cross_entropy_loss_kernel<LT> << <blocks, threads >> > (
            logits,
            targets,
            out,
            rows,
            vocab_size
            );

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss");
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    template void cuda_cross_entropy_loss<float>(const float*, const int*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss<double>(const double*, const int*, double*, size_t, size_t);








    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cross_entropy_loss_backward_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    __global__ void cross_entropy_loss_backward_kernel(const LT* logits,const int* targets,const LT* upstream,LT* grad_logits,size_t rows,size_t vocab_size) {
        size_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + (r * vocab_size);
        LT* grad_row = grad_logits + (r * vocab_size);
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        LT max_logit = row_ptr[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row_ptr[v] > max_logit) {
                max_logit = row_ptr[v];
            }
        }

        LT sum_exp = static_cast<LT>(0);
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += exp(row_ptr[v] - max_logit);
        }

        LT scale = upstream[0] / static_cast<LT>(rows);

        for (size_t v = 0; v < vocab_size; v++) {
            LT prob = exp(row_ptr[v] - max_logit) / sum_exp;
            grad_row[v] = prob * scale;
        }

        grad_row[target_id] -= scale;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss_backward
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    void cuda_cross_entropy_loss_backward(const LT* logits, const int* targets, const LT* upstream, LT* grad_logits, size_t rows, size_t vocab_size) {
        const int threads = 256;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);

        cross_entropy_loss_backward_kernel<LT> << <blocks, threads >> > (
            logits,
            targets,
            upstream,
            grad_logits,
            rows,
            vocab_size
            );

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss_backward");
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    template void cuda_cross_entropy_loss_backward<float>(const float*, const int*, const float*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss_backward<double>(const double*, const int*, const double*, double*, size_t, size_t);




    template<typename LT, int BLOCK_SIZE>
    __global__ void cross_entropy_loss_backward_kernel_fast(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size
    ) {
        size_t r = blockIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + r * vocab_size;
        LT* grad_row = grad_logits + r * vocab_size;

        int tid = threadIdx.x;
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        __shared__ LT sdata[BLOCK_SIZE];

        // 1. parallel max
        LT local_max = -INFINITY;

        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            LT x = row_ptr[v];
            if (x > local_max) local_max = x;
        }

        sdata[tid] = local_max;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                LT other = sdata[tid + stride];
                if (other > sdata[tid]) sdata[tid] = other;
            }
            __syncthreads();
        }

        LT max_logit = sdata[0];

        // 2. parallel sum exp
        LT local_sum = static_cast<LT>(0);

        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            local_sum += exp(row_ptr[v] - max_logit);
        }

        sdata[tid] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        LT sum_exp = sdata[0];
        LT scale = upstream[0] / static_cast<LT>(rows);

        // 3. parallel write grad
        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            LT prob = exp(row_ptr[v] - max_logit) / sum_exp;
            LT g = prob * scale;

            if (static_cast<int>(v) == target_id) {
                g -= scale;
            }

            grad_row[v] = g;
        }
    }

    template<typename LT>
    void cuda_cross_entropy_loss_backward_fast(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size
    ) {
        constexpr int threads = 256;
        int blocks = static_cast<int>(rows);

        cross_entropy_loss_backward_kernel_fast<LT, threads>
            << <blocks, threads >> > (
                logits,
                targets,
                upstream,
                grad_logits,
                rows,
                vocab_size
                );

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss_backward");
    }


    template void cuda_cross_entropy_loss_backward_fast<float>(const float*, const int*, const float*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss_backward_fast<double>(const double*, const int*, const double*, double*, size_t, size_t);

}