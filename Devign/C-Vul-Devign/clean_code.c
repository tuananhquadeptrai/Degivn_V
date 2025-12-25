// Clean and secure C code - no vulnerabilities
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 256

// SAFE: Uses fgets with bounds checking
int read_input_safe(char *buffer, size_t size) {
    if (buffer == NULL || size == 0) {
        return -1;
    }
    
    if (fgets(buffer, size, stdin) != NULL) {
        // Remove trailing newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        return 0;
    }
    return -1;
}

// SAFE: Uses snprintf instead of sprintf
void log_message_safe(const char *format, const char *msg) {
    char buffer[MAX_BUFFER];
    snprintf(buffer, sizeof(buffer), "%s: %s", format, msg);
    printf("%s\n", buffer);
}

// SAFE: Bounds checking before allocation
int allocate_buffer_safe(size_t size) {
    if (size == 0 || size > SIZE_MAX / 4) {
        return -1;  // Prevent integer overflow
    }
    
    size_t total = size * 4;
    char *buf = malloc(total);
    if (buf == NULL) {
        return -1;  // Check malloc result
    }
    
    memset(buf, 0, total);
    free(buf);
    return 0;
}

// SAFE: Proper memory management
char* duplicate_string(const char *src) {
    if (src == NULL) {
        return NULL;
    }
    
    size_t len = strlen(src);
    char *copy = malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }
    
    strncpy(copy, src, len);
    copy[len] = '\0';
    return copy;  // Caller must free
}

// SAFE: NULL checks before use
int process_file_safe(const char *filename) {
    if (filename == NULL) {
        return -1;
    }
    
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return -1;  // Check file open result
    }
    
    char buffer[MAX_BUFFER];
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        printf("%s", buffer);
    }
    
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    char input[MAX_BUFFER];
    
    printf("Enter text: ");
    if (read_input_safe(input, sizeof(input)) == 0) {
        log_message_safe("Input", input);
    }
    
    if (argc > 1) {
        process_file_safe(argv[1]);
    }
    
    return 0;
}
