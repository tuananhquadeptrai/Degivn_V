#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 256

int safe_read_input(char *buffer, size_t size) {
    if (buffer == NULL || size == 0) {
        return -1;
    }
    
    if (fgets(buffer, size, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        return 0;
    }
    return -1;
}

void *safe_alloc(size_t size) {
    if (size == 0 || size > 1024 * 1024) {
        return NULL;
    }
    
    void *ptr = calloc(1, size);
    return ptr;
}

void safe_free(void **ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

int safe_copy(char *dest, size_t dest_size, const char *src) {
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

int main(void) {
    char buffer[MAX_BUFFER];
    
    printf("Enter text: ");
    if (safe_read_input(buffer, sizeof(buffer)) == 0) {
        printf("You entered: %s\n", buffer);
    }
    
    char *data = safe_alloc(100);
    if (data != NULL) {
        safe_copy(data, 100, "Hello World");
        printf("Data: %s\n", data);
        safe_free((void **)&data);
    }
    
    return 0;
}
