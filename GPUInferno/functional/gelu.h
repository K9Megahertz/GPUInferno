#pragma once
#include <numeric>
#include "../util/random.h"
#include "../modules/module.h"

namespace Inferno {

	Tensor gelu(const Tensor& A);

	template <typename AT>
	void cpu_gelu(const AT* aptr, AT* optr, size_t outer, size_t dim, size_t inner, size_t off, size_t N) {



	}


}