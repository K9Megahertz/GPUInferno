#include "embedding.h"


namespace Inferno {


    /*Embedding::Embedding(size_t vocab_size, size_t embed_dim) {
        m_embeddings = Tensor::randn<float>({ vocab_size, embed_dim }, "embedding");
        register_parameter(m_embeddings);
    }


    Tensor Embedding::forward(Tensor& token_ids) {

        //TODO: dispatch for this
        //token_ids [B, T]
        //[[1,2,3,0,0,0,0],   Batch 0
        // [7,3,5,7,9,0,0],   Batch 1
        // [1,9,2,8,3,7,5]]   Batch n
        //  ^-----------^     seq_len

        bool was_1d = false;
        // m_embeddings
        // 0: [0.29, 0.76, 0.83 ... 0.59]
        // 1: [0.71, 0.33, 0.42 ... 0.99]
        //            . . .
        // n: [0.94, 0.77, 0.84 ... 0.15]
        //      ^---------------------^    embed_dim

      */



    Embedding::Embedding(size_t vocab_size, size_t embed_dim, Device device, DType dtype) {
        dispatchOne(dtype, [&](auto TagA) {
            using AT = typename decltype(TagA)::type;
            m_embeddings = Tensor::randn<AT>({ vocab_size, embed_dim }, "embedding");
        });
        register_parameter(m_embeddings); // so optimizer will see it
    }

    Tensor Embedding::forward(Tensor& token_ids) {
        // token_ids: [T] or [B, T]

        bool a_vec = (token_ids.ndim() == 1);

        Tensor token_ids_view = make_view(token_ids, token_ids.shape(), token_ids.strides(), 0, "token_ids");
        
        if (a_vec) {
            token_ids_view.shape() = { 1, token_ids_view.shape()[0] };
            token_ids_view.strides() = token_ids_view.calculate_strides(token_ids_view.shape());
        }

        Tensor out = embedding_impl(token_ids_view, m_embeddings);

        if (a_vec)
            out.shape().erase(out.shape().begin() + 0);        

        out.strides() = out.calculate_strides(out.shape());

        if (Inferno::grad_enabled) {
            GetImpl(out)->gradfn() = std::make_shared<EmbeddingBackward>(token_ids);
        }

        

        return out;
    }


    Tensor Embedding::embedding_impl(const Tensor& token_ids, Tensor& m_embeddings) {


        size_t num_batches = token_ids.shape()[0]; // B
        size_t seq_len = token_ids.shape()[1]; // T
        size_t embed_dim = m_embeddings.shape()[1];   // E
        size_t vocab_size = m_embeddings.shape()[0]; //i dont think we need this for anything


        Inferno::Tensor out(m_embeddings.dtype(), { num_batches, seq_len, embed_dim }, "embedding_out", token_ids.device());


        return dispatchTwo(m_embeddings.dtype(), token_ids.dtype(), [&](auto TagA, auto TagB) {
            using AT = typename decltype(TagA)::type;
            using BT = typename decltype(TagB)::type;

            auto tptr = GetImpl(token_ids)->data_as_ptr<BT>();
            auto eptr = GetImpl(m_embeddings)->data_as_ptr<AT>();
            auto optr = GetImpl(out)->data_as_ptr<AT>();
            
            


            switch (m_embeddings.device().m_type) {

                ////////////////////////////////////////////////////
                // CPU Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
                cpu_embedding(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                break;

                ////////////////////////////////////////////////////
                // CUDA Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
                cuda_embedding(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }

            return out;






            });


        // Create output tensor




        /*// We'll store the token ids we actually used in a flat vector [B*T]
        std::vector<int> token_ids_flat;
        token_ids_flat.reserve(num_batches * seq_len);

        // Prepare output buffer [B, T, E]
        std::vector<float> output_data;
        output_data.reserve(num_batches * seq_len * embed_dim);

        auto& embed_vec = m_embeddings.data_as<float>(); // [vocab_size * embed_dim]

        // Populate output_data and remember token ids
        for (size_t b = 0; b < num_batches; ++b) {
            for (size_t t = 0; t < seq_len; ++t) {
                int token_id = static_cast<int>(token_ids_view(b, t));
                // (optional) sanity check
                // assert(token_id >= 0 && token_id < static_cast<int>(vocab_size));

                token_ids_flat.push_back(token_id);

                size_t offset = static_cast<size_t>(token_id) * embed_dim;

                output_data.insert(
                    output_data.end(),
                    embed_vec.begin() + offset,
                    embed_vec.begin() + offset + embed_dim
                );
            }
        }




        // Only m_embeddings is a differentiable parent – token_ids are just indices
        result.setParents({ m_embeddings.m_node });

        // Backward: scatter-add from grad_out -> grad_weight
        /*result.m_node->backward_fn() = [emb_node = m_embeddings.m_node, res_node = result.m_node, token_ids_flat = std::move(token_ids_flat), num_batches, seq_len, embed_dim]() mutable
        {
            Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
                "Performing Backward of embedding");

            // If nothing propagated into this node, nothing to do
            if (!res_node->grad()) {
                return;
            }

            // Upstream gradient dL/dOutput: shape [B, T, E] or [T, E] (if squeezed later)
            auto& grad_out = res_node->grad_as<float>(); // size = B*T*E

            // Ensure grad buffer for embeddings exists and is zero-initialized
            if (!emb_node->grad()) {
                emb_node->grad() = std::static_pointer_cast<void>(std::make_shared<std::vector<float>>(emb_node->numel(), 0.0f));
            }

            auto& grad_weight = emb_node->grad_as<float>(); // [vocab_size * E]

            // Scatter-add: for each (b, t), add grad_out[b, t, :] to
            // grad_weight[token_id, :]
            const size_t BT = num_batches * seq_len;

            for (size_t i = 0; i < BT; ++i) {
                int token_id = token_ids_flat[i];

                size_t grad_offset = static_cast<size_t>(token_id) * embed_dim;
                size_t out_offset = i * embed_dim;

                for (size_t e = 0; e < embed_dim; ++e) {
                    grad_weight[grad_offset + e] += grad_out[out_offset + e];
                }
            }
        };

        // If the caller passed [T] we return [T, E] instead of [1, T, E]
        if (was_1d) {
            // again: non-mutating squeeze that returns a new Tensor
            //result = result.squeeze(0);
            result.squeeze_(0);
        }*/

    }

}