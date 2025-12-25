// Test file with known vulnerabilities
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_function(char *user_input) {
    char buffer[64];
    
    // VULNERABILITY 1: Buffer overflow - strcpy without bounds check
    strcpy(buffer, user_input);
    
    // VULNERABILITY 2: Format string vulnerability
    printf(buffer);
    
    // VULNERABILITY 3: Use after free
    char *ptr = malloc(100);
    free(ptr);
    strcpy(ptr, "use after free");
    
    // VULNERABILITY 4: No NULL check after malloc
    char *data = malloc(1024);
    memcpy(data, user_input, 1024);
    
    // VULNERABILITY 5: Gets - deprecated and dangerous
    char input[100];
    gets(input);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
