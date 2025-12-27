#include <stdio.h>
#include <string.h>

#define MAX_NAME 50
#define MAX_STUDENTS 10

typedef struct {
    int id;
    char name[MAX_NAME];
    float score;
} Student;

void init_student(Student *s, int id, const char *name, float score) {
    if (s == NULL || name == NULL) {
        return;
    }
    s->id = id;
    strncpy(s->name, name, MAX_NAME - 1);
    s->name[MAX_NAME - 1] = '\0';
    s->score = score;
}

void print_student(const Student *s) {
    if (s == NULL) {
        return;
    }
    printf("ID: %d, Name: %s, Score: %.2f\n", s->id, s->name, s->score);
}

float calculate_average(const Student *students, int count) {
    if (students == NULL || count <= 0) {
        return 0.0f;
    }
    
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += students[i].score;
    }
    return total / count;
}

int find_highest(const Student *students, int count) {
    if (students == NULL || count <= 0) {
        return -1;
    }
    
    int max_idx = 0;
    for (int i = 1; i < count; i++) {
        if (students[i].score > students[max_idx].score) {
            max_idx = i;
        }
    }
    return max_idx;
}

int main(void) {
    Student class[3];
    
    init_student(&class[0], 1, "Alice", 85.5f);
    init_student(&class[1], 2, "Bob", 92.0f);
    init_student(&class[2], 3, "Charlie", 78.5f);
    
    printf("Student List:\n");
    for (int i = 0; i < 3; i++) {
        print_student(&class[i]);
    }
    
    printf("\nAverage: %.2f\n", calculate_average(class, 3));
    
    int top = find_highest(class, 3);
    if (top >= 0) {
        printf("Top student: %s\n", class[top].name);
    }
    
    return 0;
}
