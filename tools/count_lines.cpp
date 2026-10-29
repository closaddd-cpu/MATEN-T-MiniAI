// BY MATEN/T | M/ License v1.0
// Optional streaming line counter, written in C++17; NOT a replacement for PyTorch GPU training.
#include <fstream>
#include <iostream>
#include <string>
#include <array>
int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: count_lines file.jsonl [...files]\n"; return 2; }
    for(int i=1; i<argc; ++i) {
        std::ifstream input(argv[i], std::ios::binary);
        if(!input) { std::cerr << "Cannot open: " << argv[i] << "\n"; return 1; }
        std::array<char, 1<<16> buf{};
        unsigned long long count=0;
        char last='\0';
        while(input) {
            input.read(buf.data(),buf.size());
            const auto n=input.gcount();
            for(std::streamsize j=0;j<n;j++) { if(buf[j]=='\n')count++; last=buf[j]; }
        }
        if(last!='\0' && last!='\n')count++;
        std::cout << argv[i] << "\t" << count << "\n";
    }
}
