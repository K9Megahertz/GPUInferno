#pragma once
#include "node.h"
#include "../tensor.h"
#include "../broadcastops.h"



namespace Inferno {


	class EmbeddingBackward : public Node {

	public:

		EmbeddingBackward(const Tensor& A, const Tensor& B);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_token_ids;
		Tensor m_embeddings;
		


	};



}