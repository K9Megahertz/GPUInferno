#include "cudaops.h"


namespace Inferno {

	template <typename AT, typename BT>
	__global__ void  cuda_kernel_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {


	}



	template <typename AT, typename BT>
	void cuda_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {


		constexpr int threads = 256;
		int blocks = static_cast<int>(((num_batches * seq_len * embed_dim) + threads - 1) / threads);


		cuda_kernel_embedding<AT, BT> << <blocks, threads >> > (tptr, eptr, optr, num_batches, seq_len, embed_dim);
	}


	template void cuda_embedding<int, int>(const int*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, int>(const int*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, int>(const int*, const double*, double*, size_t, size_t, size_t);

	template void cuda_embedding<int, float>(const float*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, float>(const float*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, float>(const float*, const double*, double*, size_t, size_t, size_t);

	template void cuda_embedding<int, double>(const double*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, double>(const double*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, double>(const double*, const double*, double*, size_t, size_t, size_t);


}