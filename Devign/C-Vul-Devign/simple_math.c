#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main(void) {
    int x = 10;
    int y = 20;
    
    printf("Sum: %d\n", add(x, y));
    printf("Product: %d\n", multiply(x, y));
    
    return 0;
}
