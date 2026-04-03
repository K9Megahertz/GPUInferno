#pragma once
#include <numeric>
#include "../util/random.h"
#include "../modules/module.h"

namespace Inferno {

	Tensor Softmax(Tensor& A, int axis = -1, bool keepdims = true);

	template <typename AT>
	void cpu_softmax(const AT* aptr, AT* optr, size_t outer, size_t dim, size_t inner, size_t off, size_t N) {



	}


}