#include <fstream>
#include "inferno/core/tensorimpl.h"
#include <inferno/checkpoint/checkpoint.h>
#include <inferno/util/logging.h>

namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function save old version without optim
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*void Checkpoint::save(const std::string& filename) const {


        std::ofstream out(filename, std::ios::binary);

        if (!out) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        CheckpointHeader header{};
        std::memcpy(header.magic, MAGIC, 8);
        header.version = 0;
        header.tensor_count = static_cast<uint32_t>(model.size());

        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!out) {
            throw std::runtime_error("Failed while writing checkpoint header");
        }

        for (const auto& [name, tensor] : model) {

            Tensor t = tensor;

            if (t.device().m_type != DeviceType::CPU) {
                t = t.to(Device::cpu());
            }

            if (!t.is_contiguous()) {
                //t = contiguous(t);
            }

            TensorRecordHeader trh{};
            trh.name_length = static_cast<uint32_t>(name.size());
            trh.dtype = static_cast<uint32_t>(t.dtype());
            trh.ndim = static_cast<uint32_t>(t.shape().size());
            trh.numel = static_cast<uint64_t>(t.numel());
            trh.nbytes = static_cast<uint64_t>(GetImpl(t)->nbytes());
            

            out.write(reinterpret_cast<const char*>(&trh), sizeof(trh));

            for (size_t dim : t.shape()) {
                uint64_t d = static_cast<uint64_t>(dim);
                out.write(reinterpret_cast<const char*>(&d), sizeof(d));
            }

            out.write(name.data(), trh.name_length);
            out.write(reinterpret_cast<const char*>(GetImpl(t)->raw_ptr()), trh.nbytes);

            if (!out) {
                throw std::runtime_error("Failed while writing tensor: " + name);
            }
        }

            
		out.close();


	}*/

    void Checkpoint::save(const std::string& filename) const {

        std::ofstream out(filename, std::ios::binary);

        if (!out) {            
            INFERNO_LOG_ERROR() << "Failed to open file: " << filename;
            exit(1);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        uint32_t optimizer_tensor_count = static_cast<uint32_t>(optimizer.states.size() * 2); // m + v per param

        // ------------------------------------------------------------
        // Write checkpoint header
        // ------------------------------------------------------------
        CheckpointHeader header{};
        std::memcpy(header.magic, MAGIC, 8);
        header.version = 1;
        header.tensor_count = static_cast<uint32_t>(model.size()) + optimizer_tensor_count;

        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!out) {            
            INFERNO_LOG_ERROR() << "Failed while writing checkpoint header";
            exit(1);
        }

        // ------------------------------------------------------------
        // Write training metadata
        // ------------------------------------------------------------
        TrainingMetadata metadata{};
        metadata.step = meta.step;
        metadata.total_steps = meta.total_steps;
        metadata.epoch = meta.epoch;
        metadata.total_epochs = meta.total_epochs;

        out.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));

        if (!out) {            
            INFERNO_LOG_ERROR() << "Failed while writing training metadata";
            exit(1);
        }

        // ------------------------------------------------------------
        // Write optimizer metadata
        // ------------------------------------------------------------
        struct OptimizerMetadata {
            uint64_t step;
            float lr;
            float beta1;
            float beta2;
            float eps;
            float weight_decay;
            uint64_t state_count;
        };

        OptimizerMetadata opt_meta{};
        opt_meta.step = optimizer.step;
        opt_meta.lr = optimizer.lr;
        opt_meta.beta1 = optimizer.beta1;
        opt_meta.beta2 = optimizer.beta2;
        opt_meta.eps = optimizer.eps;
        opt_meta.weight_decay = optimizer.weight_decay;
        opt_meta.state_count = static_cast<uint64_t>(optimizer.states.size());

        out.write(reinterpret_cast<const char*>(&opt_meta), sizeof(opt_meta));

        if (!out) {            
            INFERNO_LOG_ERROR() << "Failed while writing optimizer metadata";
            exit(1);
        }

        auto write_tensor_record = [&](const std::string& name, const Tensor& tensor) {
            Tensor t = tensor;

            //transfer it to the cpu if its not already there
            if (t.device().m_type != DeviceType::CPU) {
                t = t.to(Device::cpu());
            }


            //DIFNT?
            if (!t.is_contiguous()) {
                // t = contiguous(t);                
                INFERNO_LOG_ERROR() << "Cannot save non-contiguous tensor yet: " << name;
                exit(1);
            }

            // ------------------------------------------------------------
            // Write header for the tensor itself
            // ------------------------------------------------------------
            TensorRecordHeader trh{};
            trh.name_length = static_cast<uint32_t>(name.size());
            trh.dtype = static_cast<uint32_t>(t.dtype());
            trh.ndim = static_cast<uint32_t>(t.ndim());
            trh.numel = static_cast<uint64_t>(t.numel());
            trh.nbytes = static_cast<uint64_t>(GetImpl(t)->nbytes());

            out.write(reinterpret_cast<const char*>(&trh), sizeof(trh));


            //write out the shape
            for (size_t dim : t.shape()) {
                uint64_t d = static_cast<uint64_t>(dim);
                out.write(reinterpret_cast<const char*>(&d), sizeof(d));
            }

            //write out the name of the tensor
            out.write(name.data(), trh.name_length);

            //write out the actual data in the tensor
            out.write(reinterpret_cast<const char*>(GetImpl(t)->raw_ptr()), trh.nbytes);

            if (!out) {                
                INFERNO_LOG_ERROR() << "Failed while writing tensor: " << name;
                exit(1);
            }
        };

        // ------------------------------------------------------------
        // Save model tensors
        // ------------------------------------------------------------
        for (const auto& [name, tensor] : model) {
            write_tensor_record("model." + name, tensor);
        }

        // ------------------------------------------------------------
        // Save AdamW optimizer tensors
        // ------------------------------------------------------------
        for (const auto& [param_name, state] : optimizer.states) {
            write_tensor_record("optimizer." + param_name + ".m", state.m);
            write_tensor_record("optimizer." + param_name + ".v", state.v);
        }

        out.close();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function set_state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Checkpoint::set_state_dict(const StateDict& state) {
		model = state;
	}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	const StateDict& Checkpoint::state_dict() const {
		return model;
	}



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function load old version without optim
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*Checkpoint Checkpoint::load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open checkpoint file for reading: " + path);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        CheckpointHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!in) {
            throw std::runtime_error("Failed to read checkpoint header");
        }

        if (std::memcmp(header.magic, MAGIC, 8) != 0) {
            throw std::runtime_error("Invalid checkpoint file: bad magic");
        }

        if (header.version != 0) {
            throw std::runtime_error("Unsupported checkpoint version");
        }

        Checkpoint ckpt;

        for (uint32_t i = 0; i < header.tensor_count; i++) {

            TensorRecordHeader trh{};
            in.read(reinterpret_cast<char*>(&trh), sizeof(trh));

            if (!in) {
                throw std::runtime_error("Failed to read tensor record header");
            }

            std::vector<size_t> shape;
            shape.reserve(trh.ndim);

            for (uint32_t d = 0; d < trh.ndim; d++) {
                uint64_t dim = 0;
                in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

                if (!in) {
                    throw std::runtime_error("Failed to read tensor shape");
                }

                shape.push_back(static_cast<size_t>(dim));
            }

            std::string name(trh.name_length, '\0');
            if (trh.name_length > 0) {
                in.read(&name[0], trh.name_length);

                if (!in) {
                    throw std::runtime_error("Failed to read tensor name");
                }
            }

            DType dtype = static_cast<DType>(trh.dtype);

            Tensor tensor(dtype, shape, name, Device::cpu());

            void* raw = GetImpl(tensor)->raw_ptr();
            in.read(reinterpret_cast<char*>(raw), static_cast<std::streamsize>(trh.nbytes));

            if (!in) {
                throw std::runtime_error("Failed to read tensor data for: " + name);
            }

            if (tensor.numel() != trh.numel) {
                throw std::runtime_error("Tensor numel mismatch while loading: " + name);
            }

            ckpt.model[name] = tensor;
        }

        return ckpt;
    }*/

    Checkpoint Checkpoint::load(const std::string& filename) {

        std::ifstream in(filename, std::ios::binary);

        if (!in) {
            throw std::runtime_error("Failed to open checkpoint file: " + filename);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        Checkpoint ckpt;

        CheckpointHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!in) {
            throw std::runtime_error("Failed while reading checkpoint header");
        }

        if (std::memcmp(header.magic, MAGIC, 8) != 0) {
            throw std::runtime_error("Invalid checkpoint magic");
        }

        if (header.version != 1) {
            throw std::runtime_error("Unsupported checkpoint version");
        }

        // ------------------------------------------------------------
        // Read training metadata
        // ------------------------------------------------------------
        TrainingMetadata metadata{};
        in.read(reinterpret_cast<char*>(&ckpt.meta), sizeof(metadata));

        if (!in) {
            throw std::runtime_error("Failed while reading training metadata");
        }      

        // ------------------------------------------------------------
        // Read optimizer metadata
        // ------------------------------------------------------------
        struct OptimizerMetadata {
            uint64_t step;
            float lr;
            float beta1;
            float beta2;
            float eps;
            float weight_decay;
            uint64_t state_count;
        };

        OptimizerMetadata opt_meta{};
        in.read(reinterpret_cast<char*>(&opt_meta), sizeof(opt_meta));

        if (!in) {
            throw std::runtime_error("Failed while reading optimizer metadata");
        }

        ckpt.optimizer.step = static_cast<size_t>(opt_meta.step);
        ckpt.optimizer.lr = opt_meta.lr;
        ckpt.optimizer.beta1 = opt_meta.beta1;
        ckpt.optimizer.beta2 = opt_meta.beta2;
        ckpt.optimizer.eps = opt_meta.eps;
        ckpt.optimizer.weight_decay = opt_meta.weight_decay;

        // ------------------------------------------------------------
        // Helper
        // ------------------------------------------------------------
        auto starts_with = [](const std::string& s, const std::string& prefix) {
            return s.rfind(prefix, 0) == 0;
        };

        auto ends_with = [](const std::string& s, const std::string& suffix) {
            if (suffix.size() > s.size()) {
                return false;
            }

            return std::equal(
                suffix.rbegin(),
                suffix.rend(),
                s.rbegin()
            );
        };

        // ------------------------------------------------------------
        // Read tensor records
        // ------------------------------------------------------------
        for (uint32_t i = 0; i < header.tensor_count; i++) {

            TensorRecordHeader trh{};
            in.read(reinterpret_cast<char*>(&trh), sizeof(trh));

            if (!in) {
                throw std::runtime_error("Failed while reading tensor record header");
            }

            std::vector<size_t> shape;
            shape.reserve(trh.ndim);

            for (uint32_t d = 0; d < trh.ndim; d++) {
                uint64_t dim = 0;
                in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

                if (!in) {
                    throw std::runtime_error("Failed while reading tensor shape");
                }

                shape.push_back(static_cast<size_t>(dim));
            }

            std::string name;
            name.resize(trh.name_length);

            in.read(name.data(), trh.name_length);

            if (!in) {
                throw std::runtime_error("Failed while reading tensor name");
            }

            Tensor tensor(static_cast<DType>(trh.dtype), shape, name, Device::cpu());

            uint64_t expected_nbytes = static_cast<uint64_t>(GetImpl(tensor)->nbytes());

            if (expected_nbytes != trh.nbytes) {
                throw std::runtime_error("Tensor byte-size mismatch while loading tensor: " + name);
            }

            in.read(reinterpret_cast<char*>(GetImpl(tensor)->raw_ptr()), trh.nbytes);

            if (!in) {
                throw std::runtime_error("Failed while reading tensor data: " + name);
            }

            // --------------------------------------------------------
            // Route tensor into model or optimizer
            // --------------------------------------------------------

            if (starts_with(name, "model.")) {
                std::string param_name = name.substr(std::string("model.").size());

                
                ckpt.model[param_name] = tensor;
            }
            else if (starts_with(name, "optimizer.")) {
                std::string rest = name.substr(std::string("optimizer.").size());

                if (ends_with(rest, ".m")) {
                    std::string param_name = rest.substr(0, rest.size() - 2);

                    
                    ckpt.optimizer.states[param_name].m = tensor;
                }
                else if (ends_with(rest, ".v")) {
                    std::string param_name = rest.substr(0, rest.size() - 2);

                    
                    ckpt.optimizer.states[param_name].v = tensor;
                }
                else {
                    throw std::runtime_error("Unknown optimizer tensor name format: " + name);
                }
            }
            else {
                throw std::runtime_error("Unknown tensor namespace in checkpoint: " + name);
            }
        }

        return ckpt;
    }

}