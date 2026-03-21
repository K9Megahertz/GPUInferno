#pragma once
#include <vector>
#include "../tensor.h"

namespace Inferno {
    class OptimizerSGD {
        std::vector<Tensor*> params;
        float lr;

    public:
        OptimizerSGD(const std::vector<Tensor*>& parameters, float learning_rate)
            : params(parameters), lr(learning_rate) {}

        template <typename AT, typename BT>
        void cpu_step_impl(AT* dptr, BT* gptr, size_t N) {
            for (size_t i = 0; i < N; i++) {
                dptr[i] = static_cast<AT>( static_cast<double>(dptr[i]) - static_cast<double>(lr) * static_cast<double>(gptr[i]) );
            }
        }

        void step() {
            for (auto& p : params) {
                auto grad = GetImpl(*p)->grad();

                if (!grad) {
                    continue;
                }

                if (p->device() != grad->device()) {
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad device mismatch");
                    exit(1);
                }

                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Stepping on: " + p->name());

                if (p->shape() != grad->shape()) {
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad shape mismatch");
                    exit(1);
                }                

                

                dispatchTwo(p->dtype(), grad->dtype(), [&](auto TA, auto TB) {
                    using AT = typename decltype(TA)::type;
                    using BT = typename decltype(TB)::type;


                    AT* dptr = GetImpl(*p)->data_as_ptr<AT>();
                    BT* gptr = GetImpl(*grad)->data_as_ptr<BT>();

                    size_t count = p->numel();

                    if (p->device().is_cpu()) {
                        cpu_step_impl<AT, BT>(dptr, gptr, count);
                    }
                    else {
                        cuda_step_impl<AT, BT>(dptr, gptr, count, lr);
                        //Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD CUDA path not implemented");
                        //exit(1);
                    }
                    });
            }
        }

        void zero_grad() {
            for (auto& p : params) {
                p->grad() = nullptr;              
            }
        }
    };

}
