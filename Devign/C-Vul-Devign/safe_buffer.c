#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BUFFER_SIZE 128

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} SafeBuffer;

SafeBuffer *buffer_create(size_t capacity) {
    if (capacity == 0 || capacity > 1024 * 1024) {
        return NULL;
    }
    
    SafeBuffer *buf = malloc(sizeof(SafeBuffer));
    if (buf == NULL) {
        return NULL;
    }
    
    buf->data = calloc(capacity, sizeof(char));
    if (buf->data == NULL) {
        free(buf);
        return NULL;
    }
    
    buf->size = 0;
    buf->capacity = capacity;
    return buf;
}

int buffer_append(SafeBuffer *buf, const char *str) {
    if (buf == NULL || str == NULL) {
        return -1;
    }
    
    size_t len = strlen(str);
    if (buf->size + len >= buf->capacity) {
        return -1;
    }
    
    memcpy(buf->data + buf->size, str, len);
    buf->size += len;
    buf->data[buf->size] = '\0';
    return 0;
}

void buffer_destroy(SafeBuffer **buf) {
    if (buf != NULL && *buf != NULL) {
        if ((*buf)->data != NULL) {
            free((*buf)->data);
            (*buf)->data = NULL;
        }
        free(*buf);
        *buf = NULL;
    }
}

int safe_read_line(char *buffer, size_t size, FILE *fp) {
    if (buffer == NULL || size == 0 || fp == NULL) {
        return -1;
    }
    
    if (fgets(buffer, (int)size, fp) == NULL) {
        return -1;
    }
    
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
        buffer[len - 1] = '\0';
    }
    return 0;
}

int main(void) {
    SafeBuffer *buf = buffer_create(BUFFER_SIZE);
    if (buf == NULL) {
        fprintf(stderr, "Failed to create buffer\n");
        return 1;
    }
    
    if (buffer_append(buf, "Hello, ") == 0) {
        buffer_append(buf, "World!");
    }
    
    printf("Buffer: %s\n", buf->data);
    printf("Size: %zu\n", buf->size);
    
    buffer_destroy(&buf);
    return 0;
}
