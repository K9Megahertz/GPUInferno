#include <inferno/core/tensor.h>
#include <inferno/loss/crossentropyloss.h>
#include "inferno/gradfn/crossentropylossbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/cpuops.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/core/dtype_dispatch.h"










namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function forward
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor CrossEntropyLoss::forward(Tensor& logits, Tensor& target) {
        if (logits.device() != target.device()) {
            INFERNO_LOG_ERROR() << "Incompatible device types on tensor parameters in cross_entropy_loss" << std::endl;
            exit(1);
        }

        if (logits.ndim() < 2) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss requires logits rank >= 2" << std::endl;
            exit(1);
        }

        if (target.ndim() != logits.ndim() - 1) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss requires target rank = logits rank - 1" << std::endl;
            exit(1);
        }

        // logits shape [..., V]
        // target shape [...]
        for (size_t i = 0; i < target.ndim(); i++) {
            if (target.shape()[i] != logits.shape()[i]) {
                INFERNO_LOG_ERROR() << "Shape mismatch on tensor parameters in cross_entropy_loss" << std::endl;
                exit(1);
            }
        }

        if (target.dtype() != DType::Int32) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss currently requires target dtype = Int32" << std::endl;
            exit(1);
        }

        if (!(logits.dtype() == DType::Float32 || logits.dtype() == DType::Float64)) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss currently requires logits dtype = Float32 or Float64" << std::endl;
            exit(1);
        }

        // Optional safety check if your raw kernels assume contiguous memory.
        if (!logits.is_contiguous()) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss currently requires contiguous logits" << std::endl;
            exit(1);
        }

        if (!target.is_contiguous()) {
            INFERNO_LOG_ERROR() << "cross_entropy_loss currently requires contiguous target" << std::endl;
            exit(1);
        }

        size_t rows = target.numel();
        size_t vocab_size = logits.shape().back();

        auto implTarget = GetImpl(target);
        const int* tptr = implTarget->data_as_ptr<int>();

        return dispatchFloat(logits.dtype(), [&](auto TLogits) {
            using LT = typename decltype(TLogits)::type;

            auto implLogits = GetImpl(logits);
            const LT* lptr = implLogits->data_as_ptr<LT>();

            Tensor out(dtype_of_v<LT>, std::vector<size_t>{1}, "cross_entropy_loss", logits.device(), true);
            auto implOut = GetImpl(out);
            LT* optr = implOut->data_as_ptr<LT>();

            switch (logits.device().m_type) {
            case DeviceType::CPU:
                INFERNO_LOG_DEBUG() << "CPU Code path - Using normal cross_entropy_loss path" << std::endl;
                cpu_cross_entropy_loss(lptr, tptr, optr, rows, vocab_size);
                break;

            case DeviceType::CUDA:
                INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal cross_entropy_loss path" << std::endl;
                cuda_cross_entropy_loss(lptr, tptr, optr, rows, vocab_size);
                break;

            default:
                INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
                exit(1);
            }

            if ((Inferno::grad_enabled) && (logits.requires_grad())) {
                INFERNO_LOG_DEBUG() << "CrossEntropyLoss - Making a CrossEntropyLossBackward node" << std::endl;
                implOut->gradfn() = std::make_shared<CrossEntropyLossBackward>(logits, target);
            }

            return out;
            });
    }

}