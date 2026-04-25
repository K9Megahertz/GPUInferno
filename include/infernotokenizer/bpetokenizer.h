#pragma once

#include <vector>
#include <string>
#include <cstdint>



namespace InfernoTokenizer {

    struct TokenizerConfig {
        std::string merges_file;
        std::string vocab_file;
    };

    class BPETokenizer {
    public:

        bool Initialize(const TokenizerConfig& config);

        std::vector<uint32_t> encode(const std::string& text);
        std::string decode(const std::vector<uint32_t>& tokens);
        

        void load_merges(const std::string& file);
        void load_vocab(const std::string& file);
    };

}