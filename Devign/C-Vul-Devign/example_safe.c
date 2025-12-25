/**
 * Example of secure C code - demonstrates safe coding practices
 * This file should be detected as NOT vulnerable
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 128

// Safe string copy with bounds checking
int safe_copy(char *dest, size_t dest_size, const char *src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }
    
    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        src_len = dest_size - 1;
    }
    
    memcpy(dest, src, src_len);
    dest[src_len] = '\0';
    return 0;
}

// Safe integer addition with overflow check
int safe_add(int a, int b, int *result) {
    if (result == NULL) {
        return -1;
    }
    
    if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b)) {
        return -1;  // Overflow detected
    }
    
    *result = a + b;
    return 0;
}

// Safe memory allocation with size check
void* safe_malloc(size_t size) {
    if (size == 0 || size > (1024 * 1024 * 100)) {  // Max 100MB
        return NULL;
    }
    
    void *ptr = malloc(size);
    if (ptr != NULL) {
        memset(ptr, 0, size);  // Initialize to zero
    }
    return ptr;
}

// Safe file reading with proper error handling
int read_file_safe(const char *filename, char *buffer, size_t buf_size) {
    if (filename == NULL || buffer == NULL || buf_size == 0) {
        return -1;
    }
    
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return -1;
    }
    
    size_t bytes_read = fread(buffer, 1, buf_size - 1, fp);
    buffer[bytes_read] = '\0';
    
    fclose(fp);
    return (int)bytes_read;
}

int main(int argc, char *argv[]) {
    char buffer[BUFFER_SIZE];
    
    // Safe input handling
    if (argc > 1) {
        if (safe_copy(buffer, sizeof(buffer), argv[1]) == 0) {
            printf("Input: %s\n", buffer);
        }
    }
    
    // Safe memory usage
    int *numbers = safe_malloc(10 * sizeof(int));
    if (numbers != NULL) {
        for (int i = 0; i < 10; i++) {
            numbers[i] = i * 2;
        }
        free(numbers);
    }
    
    return 0;
}
