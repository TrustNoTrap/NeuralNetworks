#ifndef LOG_HPP
#define LOG_HPP

// This will help us log data to the FS
#include <iostream>
#include <string>
#include <fstream>

// Named L to avoid conflicts
namespace L {
    void log(const std::string& msg, const std::string& filename = "default.log") {
        // ofstream -> output file stream, std::ios::app -> append
        std::ofstream file(filename, std::ios::app); // Append mode

        if (file.is_open()) {
            file << msg;
            file.close();

        } else {
            std::cerr << "Unable to open log file : " << filename << std::endl;
        }
    }
}


#endif