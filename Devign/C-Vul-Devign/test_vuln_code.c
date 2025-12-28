/* Vulnerable Code - Buffer Overflow & Use After Free */
/* For CI/CD Testing */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void buffer_overflow(char *input) {
    char buffer[16];
    strcpy(buffer, input);  /* VULN: No bounds check */
    printf("Data: %s\n", buffer);
}

char* use_after_free() {
    char *ptr = malloc(32);
    strcpy(ptr, "secret data");
    free(ptr);
    return ptr;  /* VULN: Returning freed pointer */
}

void format_string(char *user_input) {
    printf(user_input);  /* VULN: Format string */
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        buffer_overflow(argv[1]);
        format_string(argv[1]);
    }
    
    char *data = use_after_free();
    printf("%s\n", data);  /* VULN: Use after free */
    
    return 0;
}
