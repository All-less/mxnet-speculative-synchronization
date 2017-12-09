#include <iostream>
#include <fstream>

/**
 * This program read binary files generated from
 * GloVe (https://github.com/stanfordnlp/GloVe) and
 * output text representation to screen.
 */
int main(void) {
    std::ifstream istream("cooccurrence.shuf.bin", std::ios::binary);

    int x, y;
    double d;

    while (!istream.eof()) {

        istream.read(reinterpret_cast<char*>(&x), sizeof x);
        istream.read(reinterpret_cast<char*>(&y), sizeof y);
        istream.read(reinterpret_cast<char*>(&d), sizeof d);

        std::cout << x << " " << y << " " << d << std::endl;

    }

    return 0;
}
