/**
 * Safe Calculator - Demonstrates secure C programming practices
 * This file contains no vulnerabilities (buffer overflow, integer overflow, etc.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_INPUT_SIZE 32

bool safe_add(int a, int b, int *result) {
    if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b)) {
        return false;
    }
    *result = a + b;
    return true;
}

bool safe_multiply(int a, int b, int *result) {
    if (a > 0 && b > 0 && a > INT_MAX / b) return false;
    if (a > 0 && b < 0 && b < INT_MIN / a) return false;
    if (a < 0 && b > 0 && a < INT_MIN / b) return false;
    if (a < 0 && b < 0 && a < INT_MAX / b) return false;
    *result = a * b;
    return true;
}

bool safe_divide(int a, int b, int *result) {
    if (b == 0) {
        return false;
    }
    if (a == INT_MIN && b == -1) {
        return false;
    }
    *result = a / b;
    return true;
}

void print_result(const char *operation, int a, int b, int result) {
    printf("%d %s %d = %d\n", a, operation, b, result);
}

int main(void) {
    int a = 100;
    int b = 25;
    int result = 0;

    printf("Safe Calculator Demo\n");
    printf("====================\n");

    if (safe_add(a, b, &result)) {
        print_result("+", a, b, result);
    }

    if (safe_multiply(a, b, &result)) {
        print_result("*", a, b, result);
    }

    if (safe_divide(a, b, &result)) {
        print_result("/", a, b, result);
    }

    return 0;
}
