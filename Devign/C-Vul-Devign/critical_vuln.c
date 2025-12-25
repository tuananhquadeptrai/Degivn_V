// Critical vulnerabilities test file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// VULNERABILITY: Buffer overflow with gets()
void read_input() {
    char buffer[32];
    gets(buffer);  // DANGEROUS: No bounds checking
    printf("You entered: %s\n", buffer);
}

// VULNERABILITY: Format string attack
void log_message(char *msg) {
    printf(msg);  // DANGEROUS: Format string vulnerability
}

// VULNERABILITY: Integer overflow leading to heap overflow
void allocate_buffer(int size) {
    int total = size * 4;  // Can overflow
    char *buf = malloc(total);
    memset(buf, 0, size * 4);  // Use original calculation
}

// VULNERABILITY: Double free
void process_data(char *data) {
    char *copy = malloc(strlen(data) + 1);
    strcpy(copy, data);
    free(copy);
    // ... some code ...
    free(copy);  // DANGEROUS: Double free
}

// VULNERABILITY: Use after free
char* get_string() {
    char *str = malloc(100);
    strcpy(str, "Hello");
    free(str);
    return str;  // DANGEROUS: Returning freed memory
}

// VULNERABILITY: NULL pointer dereference
void process_file(FILE *fp) {
    char buffer[256];
    fgets(buffer, 256, fp);  // No NULL check on fp
}

int main() {
    read_input();
    return 0;
}
