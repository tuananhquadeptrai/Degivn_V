/**
 * Safe Array Operations - No vulnerabilities
 * Used for CI/CD testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 100

int safe_array_sum(const int *arr, size_t len) {
    if (arr == NULL || len == 0) {
        return 0;
    }
    int sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

int safe_array_max(const int *arr, size_t len) {
    if (arr == NULL || len == 0) {
        return 0;
    }
    int max = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void print_array(const int *arr, size_t len) {
    if (arr == NULL) {
        printf("NULL array\n");
        return;
    }
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%d", arr[i]);
        if (i < len - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main(void) {
    int numbers[] = {10, 25, 8, 42, 15};
    size_t len = sizeof(numbers) / sizeof(numbers[0]);

    printf("=== Safe Array Demo ===\n");
    printf("Array: ");
    print_array(numbers, len);
    printf("Sum: %d\n", safe_array_sum(numbers, len));
    printf("Max: %d\n", safe_array_max(numbers, len));

    return 0;
}
