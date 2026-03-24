#include "embeddingbackward.h"


namespace Inferno {


	EmbeddingBackward::EmbeddingBackward(const Tensor& A) : m_A(A) {


	}

	void EmbeddingBackward::backward() {

		NoGradGuard guard;
		// upstream gradient dL/d(output)
		Tensor g_out = Engine::grad_in(this, 0);

		// for add: dL/dA = g_out, dL/dB = g_out		
		
		//Tensor g_a = sum_to_shape();

		// find parent nodes
		//auto na = GetImpl(m_A)->grad_edge();
		

		// send gradients upstream
		//if (na)
//			Engine::accumulate(na.get(), 0, g_a);

		


	}

	void EmbeddingBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};

	}

	void EmbeddingBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
		
	}


}