/**
 * Vulnerable Code Example - Buffer Overflow
 * This file contains intentional vulnerabilities for testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vulnerable_copy(char *input) {
    char buffer[32];
    strcpy(buffer, input);  // VULNERABLE: No bounds checking
    printf("Copied: %s\n", buffer);
}

void format_string_vuln(char *user_input) {
    printf(user_input);  // VULNERABLE: Format string vulnerability
}

char* use_after_free() {
    char *ptr = malloc(64);
    strcpy(ptr, "Hello");
    free(ptr);
    return ptr;  // VULNERABLE: Use after free
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_copy(argv[1]);
        format_string_vuln(argv[1]);
    }
    
    char *data = use_after_free();
    printf("%s\n", data);  // VULNERABLE: Dereferencing freed memory
    
    return 0;
}

// Test trigger
// Trigger annotation test
