#include "selectbackward.h"

namespace Inferno {

	SelectBackward::SelectBackward(const Tensor& A, int axis, size_t index)
		: m_A(A), m_axis(axis), m_index(index) {
		set_name("SelectBackward");

	}

	void SelectBackward::backward() {
		NoGradGuard guard;
		Tensor g_out = Engine::grad_in(this, 0);

		Tensor g_a = dispatchTwo(m_A.dtype(), g_out.dtype(), [&](auto TA, auto TG) {
			using AT = typename decltype(TA)::type;
			using GT = typename decltype(TG)::type;
			using RT = promote_t<AT, GT>;

			Tensor out(dtype_of_v<RT>, m_A.shape(), "SelectBackward", m_A.device());			
			RT* optr = GetImpl(out)->data_as_ptr<RT>();

			// zero initialize
			switch (m_A.device().m_type) {
			////////////////////////////////////////////////////
			// CPU Code Path
			////////////////////////////////////////////////////
			case DeviceType::CPU: {
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal select backward path");
				for (size_t i = 0; i < out.numel(); ++i)
					optr[i] = static_cast<RT>(0);
				break;
			}

			////////////////////////////////////////////////////
			// CUDA Code Path
			////////////////////////////////////////////////////
			case DeviceType::CUDA:				
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal select backward path");
				Inferno::cuda_fill(optr, RT(0), out.numel());
				break;
			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			GT* gptr = GetImpl(g_out)->data_as_ptr<GT>();			

			switch (g_out.device().m_type) {
			case DeviceType::CPU:
				cpu_select_backward_strided<GT, RT>(
					gptr,
					optr,
					g_out.shape(),
					g_out.strides(),
					m_A.strides(),
					GetImpl(g_out)->offset(),
					GetImpl(out)->offset(),
					m_axis,
					m_index
				);
				break;

			case DeviceType::CUDA:
				cuda_select_backward_strided<GT, RT>(
					gptr,
					optr,
					g_out.shape(),
					g_out.strides(),
					m_A.strides(),
					GetImpl(g_out)->offset(),
					GetImpl(out)->offset(),
					m_axis,
					m_index
				);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			return out;
			});

		auto na = GetImpl(m_A)->grad_edge();
		if (na)
			Engine::accumulate(na.get(), 0, g_a);
	}

	void SelectBackward::release() {
		m_A = Tensor{};
	}

	void SelectBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
	}



}