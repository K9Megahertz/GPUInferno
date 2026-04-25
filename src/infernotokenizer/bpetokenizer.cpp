#include <infernotokenizer/bpetokenizer.h>

namespace InfernoTokenizer {

	
	bool BPETokenizer::Initialize(const TokenizerConfig& config) {




		return true;
	}

	std::vector<uint32_t> BPETokenizer::encode(const std::string& text) {

		std::vector<uint32_t> ret{};

		return ret;

	}
	std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) {

		std::string ret;

		return ret;

	}

}