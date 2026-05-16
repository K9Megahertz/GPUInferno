#include <inferno/functional/gelu.h>
#include "inferno/functional/gelu_kernels.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/tensorimpl.h"
#include "inferno/gradfn/gelubackward.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function gelu
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor gelu(const Tensor& A) {

		return dispatchAny(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			using RT = promote_t<AT, float>;
					

			Inferno::Tensor out(dtype_of_v<RT>, A.shape(), "gelu", A.device(), true);

			auto implA = GetImpl(A);
			auto implout = GetImpl(out);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto optr = implout->data_as_ptr<RT>();


			std::vector<size_t> shape = implA->shape();
			std::vector<size_t> astrides = implA->strides();
			size_t aoffset = implA->offset();
			std::vector<size_t> ostrides = implout->strides();
			size_t ooffset = implout->offset();		


			const size_t N = A.numel();
			const size_t off = A.offset();

			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:				
				if (A.is_contiguous() && out.is_contiguous()) {
					INFERNO_LOG_DEBUG() << "CPU Code path - Using optimized gulu path" << std::endl;
					cpu_gelu<AT,RT>(aptr, optr, N, aoffset);
				}
				else {
					INFERNO_LOG_DEBUG() << "CPU Code path - Using strided gulu path" << std::endl;
					cpu_gelu_strided<AT, RT>(aptr, optr, shape, astrides, ostrides, aoffset, ooffset);
				}
				
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:				
				if (A.is_contiguous() && out.is_contiguous()) {
					INFERNO_LOG_DEBUG() << "CUDA Code path - Using optimized gelu path" << std::endl;
					cuda_gelu<AT, RT>(aptr, optr, N, aoffset);
					
				}
				else {
					INFERNO_LOG_DEBUG() << "CUDA Code path - Using strided gelu path" << std::endl;
					cuda_gelu_strided<AT, RT>(aptr, optr, shape, astrides, ostrides, aoffset, ooffset);
				}				
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Gelu - Making an GeluBackward node" << std::endl;
				implout->gradfn() = std::make_shared<GeluBackward>(A, out);
			}


			return out;
			});

	}


}