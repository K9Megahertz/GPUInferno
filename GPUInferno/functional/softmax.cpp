#include "softmax.h"
#include "../cuda/cudaops.h"


namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function softmax
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor Softmax(Tensor& A, int axis, bool keepdims) {


		return dispatchOne(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			//using RT = promote_t<AT, float>;

			const size_t ndim = A.ndim();

			//adjust negative axis specification
			int ax = (axis < 0) ? int(ndim) + axis : axis;

			//valid axis?
			if (ax < 0 || ax >= int(ndim)) {
				throw std::runtime_error("softmax: invalid axis");
			}

			Inferno::Tensor out(dtype_of_v<AT>, A.shape(), "softmax", A.device());

			//get pointers to data
			auto aptr = GetImpl(A)->data_as_ptr<AT>();			
			auto optr = GetImpl(out)->data_as_ptr<AT>();

			// compute outer, axis, inner dimensions
			size_t outer = std::accumulate(A.shape().begin(), A.shape().begin() + ax, 1, std::multiplies<size_t>());
			size_t dim = A.shape()[ax];
			size_t inner = std::accumulate(A.shape().begin() + ax + 1, A.shape().end(), 1, std::multiplies<size_t>());

			const size_t N = A.numel();
			const size_t off = A.offset();

			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_softmax(aptr, optr, outer, dim, inner, off, N);
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				//cuda_softmax(aptr, optr, outer, dim, inner, off, N);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad() || out.requires_grad())) {
				//implout->gradfn() = std::make_shared<SoftmaxBackward>(A, out);
			}


			return out;
			});

	}


}