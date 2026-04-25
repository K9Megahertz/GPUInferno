#include <unordered_map>
#include <memory>
#include <string>



namespace Tokenizer {

    class PreTokenizer {

    public:

        PreTokenizer();
        ~PreTokenizer();

        PreTokenizer(const PreTokenizer&) = delete;
        PreTokenizer& operator=(const PreTokenizer&) = delete;


        // Process a file from disk.
        //std::unordered_map<std::string, uint64_t> process_file(const std::string& filename);
        std::vector<std::string> split(const std::string& text);

    private:       
     
        class Impl;
        std::unique_ptr<Impl> m_impl;

    };


}