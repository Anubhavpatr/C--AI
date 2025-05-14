#include "Logger.h"
// cannot initializa in a header file
std::string Logger::file_path = "";
bool Logger::set = false;
Logger::Loggermode Logger::current_mode = Logger::Loggermode::NONE;