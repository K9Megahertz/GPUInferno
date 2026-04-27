
#include <inferno/optim/adamw.h>
#include <inferno/gradengine/engine.h>
#include <inferno/core/dtype_dispatch.h>
#include "inferno/cuda/cudaops.h"

namespace Inferno {


    void OptimizerAdamW::step() {
        NoGradGuard guard;

        m_step++;

        float bias_correction1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_step));
        float bias_correction2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_step));

        for (Tensor* p : m_parameters) {
            if (p == nullptr || !p->grad()) {
                continue;
            }

            Tensor* grad = p->grad().get();

            AdamWState& state = m_state[p];

            if (!state.initialized) {
                state.m = Tensor::zeros_like(*p);
                state.v = Tensor::zeros_like(*p);
                state.initialized = true;
            }

            dispatchFloatTwo(p->dtype(), grad->dtype(), [&](auto TA, auto TB) {
                using AT = typename decltype(TA)::type;
                using BT = typename decltype(TB)::type;

                AT* pptr = GetImpl(*p)->data_as_ptr<AT>();
                BT* gptr = GetImpl(*grad)->data_as_ptr<BT>();

                AT* mptr = GetImpl(state.m)->data_as_ptr<AT>();
                AT* vptr = GetImpl(state.v)->data_as_ptr<AT>();

                size_t count = p->numel();

                switch (p->device().m_type) {
                case DeviceType::CPU:
                    cpu_adamw_step_impl<AT, BT>(
                        pptr,
                        gptr,
                        mptr,
                        vptr,
                        count,
                        m_lr,
                        m_beta1,
                        m_beta2,
                        m_eps,
                        m_weight_decay,
                        bias_correction1,
                        bias_correction2
                    );
                    break;

                case DeviceType::CUDA:
                    cuda_adamw_step_impl<AT, BT>(
                        pptr,
                        gptr,
                        mptr,
                        vptr,
                        count,
                        m_lr,
                        m_beta1,
                        m_beta2,
                        m_eps,
                        m_weight_decay,
                        bias_correction1,
                        bias_correction2
                    );
                    break;

                default:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                    std::exit(1);
                }
                });
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function zero_grad
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void OptimizerAdamW::zero_grad() {
        for (auto& p : m_parameters) {
            if (p != nullptr) {
                p->grad() = nullptr;
            }
        }
    }



}