/**
 * Safe C code example - No vulnerabilities
 * This file demonstrates secure coding practices
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 256

/**
 * Safely copies a string with bounds checking
 */
int safe_string_copy(char *dest, size_t dest_size, const char *src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }
    
    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        return -1;
    }
    
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
    
    return 0;
}

/**
 * Safely allocates and initializes memory
 */
int *safe_array_alloc(size_t count) {
    if (count == 0 || count > 1000000) {
        return NULL;
    }
    
    int *arr = calloc(count, sizeof(int));
    if (arr == NULL) {
        return NULL;
    }
    
    return arr;
}

/**
 * Safely accesses array with bounds checking
 */
int safe_array_get(const int *arr, size_t size, size_t index, int *value) {
    if (arr == NULL || value == NULL) {
        return -1;
    }
    
    if (index >= size) {
        return -1;
    }
    
    *value = arr[index];
    return 0;
}

int main(void) {
    char buffer[BUFFER_SIZE];
    const char *message = "Hello, secure world!";
    
    if (safe_string_copy(buffer, sizeof(buffer), message) == 0) {
        printf("Message: %s\n", buffer);
    }
    
    int *numbers = safe_array_alloc(10);
    if (numbers != NULL) {
        for (size_t i = 0; i < 10; i++) {
            numbers[i] = (int)(i * 2);
        }
        
        int value;
        if (safe_array_get(numbers, 10, 5, &value) == 0) {
            printf("Value at index 5: %d\n", value);
        }
        
        free(numbers);
        numbers = NULL;
    }
    
    return 0;
}
