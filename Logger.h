#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>

class Logger
{
    static std::string file_path;
    static bool set;
    public:
        enum class Loggermode { NONE, OPTIMIZED, DEBUG, INFO, ERROR};
    // set the mode

    private:
        static Loggermode current_mode;

    public:
        static void basicConfig(std::string file_path_,Loggermode mode)
        {
            if(!set)
            {
                file_path = file_path_;
                std::ofstream ofs(file_path_,std::ios::out);
                ofs.close();
                current_mode = mode;
                set = true;
            }
        }

        static void info(std::string line)
        {
            if(current_mode != Loggermode::OPTIMIZED)
            {
                std::ofstream ofs(file_path,std::ios::app); // append mode
                if(ofs.is_open())
                {
                    ofs << line << std::endl;
                    ofs.close();
                }
            }
        }

        static void error(std::string line)
        {
            std::ofstream ofs(file_path,std::ios::app); // append mode
            if(ofs.is_open())
            {
                ofs << "Error: " << line << std::endl;
                ofs.close();
            }
        }
};

#endif