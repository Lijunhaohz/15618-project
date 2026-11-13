#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Exposure fusion skeleton\n"
                  << "Image I/O, command-line options, and fusion are not implemented yet.\n";
        return 0;
    }

    std::cerr << "Error: image I/O, command-line options, and fusion are not implemented yet.\n";
    return 1;
}
