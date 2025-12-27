#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 256

int safe_copy(char *dest, size_t dest_size, const char *src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
    return 0;
}

int main(void) {
    char buffer[BUFFER_SIZE];
    const char *message = "Hello, safe world!";
    
    if (safe_copy(buffer, sizeof(buffer), message) == 0) {
        printf("Message: %s\n", buffer);
    }
    
    return 0;
}
