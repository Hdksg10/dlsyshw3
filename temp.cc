#ifndef OPERATOPMS
#define OPERATOPMS
#define BINARY_OP(OP_NAME, a, b) \
    OP_NAME(a, b)
#define ADD(a, b) ((a) + (b))
#endif



#include <iostream>

using namespace std;

int main() {
    cout << BINARY_OP(ADD, 1, 2) << endl;
    return 0;
}