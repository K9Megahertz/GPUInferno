#include <iostream>
#include "bpetrainer.h"




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function main()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[]) {


    std::string input_file;
    std::string merges_file;
    std::string vocab_file;
    size_t vocab_size = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];


        //input file (this is the corpus of text you want to train the BPETokenizer on)
        if (arg == "-i") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -i\n";
                return 1;
            }
            input_file = argv[++i];
        }        
        
        //merges output file (this file will contain a list of all the merges)
        else if (arg == "-m") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -m\n";
                return 1;
            }
            merges_file = argv[++i];
        }

        //vocab output file (this will be your vocabulary)
        else if (arg == "-v") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -v\n";
                return 1;
            }
            vocab_file = argv[++i];
        }

        //number of merges (this will be your vocabulary size and then 256 + special token count will be added)
        else if (arg == "-s") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -s\n";
                return 1;
            }
            vocab_size = std::stoull(argv[++i]);
        }

        //covfefe?
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    // Validate required args
    if (input_file.empty() || merges_file.empty() || vocab_file.empty() || vocab_size == 0) {
        std::cerr << "Usage: -i <input> -m <merges> -v <vocab> -s <size>\n";
        return 1;
    }


	
	BPETrainer trainer;

	BPETrainerConfig config;
    config.input_file = input_file;	
    config.mergerules_output_file = merges_file;
    config.vocab_output_file = vocab_file;
	config.target_vocab_size = vocab_size;
    config.initial_token_count = 256;
    config.special_tokens = {
       "<|endoftext|>",
       "<|user|>",
       "<|assistant|>"
    };

	trainer.train(config);
    trainer.save();





	return 0;
}




