#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void process_input() {
    char buffer[64];
    gets(buffer);
    printf(buffer);
}

char *read_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    char *data = malloc(1024);
    fread(data, 1, 1024, fp);
    fclose(fp);
    return data;
}

void copy_data(char *input) {
    char dest[32];
    strcpy(dest, input);
    printf("Copied: %s\n", dest);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        copy_data(argv[1]);
        process_input();
    }
    return 0;
}
