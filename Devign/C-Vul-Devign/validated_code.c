#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 64

int validate_input(const char *input) {
    if (input == NULL) {
        return 0;
    }
    size_t len = strlen(input);
    if (len == 0 || len >= MAX_LEN) {
        return 0;
    }
    return 1;
}

char *safe_duplicate(const char *src) {
    if (src == NULL) {
        return NULL;
    }
    
    size_t len = strlen(src);
    char *copy = malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }
    
    memcpy(copy, src, len);
    copy[len] = '\0';
    return copy;
}

void print_message(const char *msg) {
    if (msg != NULL) {
        printf("%s\n", msg);
    }
}

int main(int argc, char *argv[]) {
    const char *text = "Hello Safe World";
    
    if (validate_input(text)) {
        char *dup = safe_duplicate(text);
        if (dup != NULL) {
            print_message(dup);
            free(dup);
        }
    }
    
    return 0;
}
