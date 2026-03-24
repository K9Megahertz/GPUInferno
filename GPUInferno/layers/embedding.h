#pragma once
#include <vector>
#include "module.h"
#include "../tensor.h"
#include "../GradFN/embeddingbackward.h"


namespace Inferno {

    class Embedding : public Inferno::Module {

    public:

        Embedding(size_t vocab_size, size_t embed_dim, Device device = Inferno::Device::cpu(), DType dtype = Inferno::DType::Float32);
        Tensor forward(Tensor& token_ids) override;
        Tensor embedding_impl(const Tensor& token_ids, Tensor& m_embeddings);

        Tensor operator()(Tensor& input) {
            return forward(input);
        }
    private:

        Tensor m_embeddings;  // [vocab_size, embedding_dim]
    };

    template <typename AT, typename BT>
    void cpu_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {


    }

}