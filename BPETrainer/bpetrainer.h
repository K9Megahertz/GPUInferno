#pragma once
#include <string>
#include <pretokenizer/pretokenizer.h>
#include <map>

using PairMap = std::unordered_map<uint64_t, uint64_t>;

struct CorpusEntry {
	std::vector<uint32_t> symbols;
	uint64_t freq;
};

struct MergeRule {
	uint32_t first;
	uint32_t second;
};

struct BPETrainerConfig {
	std::string input_file;
	std::string mergerules_output_file;
	std::string vocab_output_file;

	uint32_t target_vocab_size;
	uint32_t initial_token_count;	
	std::vector<std::string> special_tokens;
};


class BPETrainer {


public:

	void train(const BPETrainerConfig& config);
	void save();

private:

	std::unordered_map<std::string, uint64_t> process_file(std::string& filename);
	PairMap build_pairmap_from_corpus(std::vector<CorpusEntry>& corpus);
	std::vector<CorpusEntry> build_corpus_from_piece_freqs(const std::unordered_map<std::string, uint64_t>& pf);
	uint64_t merge_best_pair_optimized(std::vector<CorpusEntry>& corpus, PairMap& pairmap, std::vector<MergeRule>& rules);
	std::vector<uint32_t> concat(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);

	void process_special_aware_text(const std::string& text, std::unordered_map<std::string, uint64_t>& piece_map);
	bool find_special_at(const std::string& text, size_t pos, std::string& matched) const;
	void add_normal_text(const std::string& text, std::unordered_map<std::string, uint64_t>& piece_map);

	uint32_t m_newtokenid = 256;
	BPETrainerConfig m_config;
	std::vector<MergeRule> m_rules;
	std::map<uint32_t, std::vector<uint32_t>> m_vocab;
	Tokenizer::PreTokenizer m_tokenizer;


};
