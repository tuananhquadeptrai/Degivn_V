/**
 * Demo Safe Code - No vulnerabilities
 * Used for CI/CD testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BUFFER_SIZE 256

typedef struct {
    int id;
    char name[64];
    double score;
} Student;

int safe_string_copy(char *dest, size_t dest_size, const char *src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }
    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        return -1;
    }
    memcpy(dest, src, src_len + 1);
    return 0;
}

int safe_add(int a, int b, int *result) {
    if (result == NULL) {
        return -1;
    }
    if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b)) {
        return -1;
    }
    *result = a + b;
    return 0;
}

void init_student(Student *s, int id, const char *name, double score) {
    if (s == NULL || name == NULL) {
        return;
    }
    s->id = id;
    safe_string_copy(s->name, sizeof(s->name), name);
    s->score = score;
}

void print_student(const Student *s) {
    if (s == NULL) {
        printf("Invalid student\n");
        return;
    }
    printf("ID: %d, Name: %s, Score: %.2f\n", s->id, s->name, s->score);
}

int main(void) {
    Student student;
    int sum = 0;

    printf("=== Safe Code Demo ===\n");

    init_student(&student, 1, "Alice", 95.5);
    print_student(&student);

    if (safe_add(100, 200, &sum) == 0) {
        printf("Sum: %d\n", sum);
    }

    return 0;
}
