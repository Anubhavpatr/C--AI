#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <deque>
#include <string>
#include <stdexcept>

std::vector<std::string> read_file(std::string filename)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> words;

    if(!file)
    {
        throw std::runtime_error("Failed to open names.txt");
    }

    while(std::getline(file,line))
    {
        words.push_back(line);
    }

    return words;
}

std::tuple<std::vector<std::vector<int>>,std::vector<int>> build_dataset(std::vector<std::string>& words,int block_size,std::map<char,int>& encoder)
{   
    std::vector<std::vector<int>> X;
    std::vector<int> Y;

    for(const auto& w : words)
    {
        std::deque<int> context(block_size);
        for(const auto& ch : w)
        {
            int ix = encoder[ch];
            std::vector<int> temp(context.begin(),context.end());
            X.push_back(temp);
            Y.push_back(ix);
            context.pop_front();
            context.push_back(ix);
        }
    }
    return std::make_tuple(X,Y);
} 


