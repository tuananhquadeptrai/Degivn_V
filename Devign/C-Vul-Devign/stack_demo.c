#include <stdio.h>
#include <string.h>

#define MAX_SIZE 100

typedef struct {
    int items[MAX_SIZE];
    int top;
} Stack;

void stack_init(Stack *s) {
    if (s != NULL) {
        s->top = -1;
    }
}

int stack_push(Stack *s, int value) {
    if (s == NULL || s->top >= MAX_SIZE - 1) {
        return -1;
    }
    s->items[++s->top] = value;
    return 0;
}

int stack_pop(Stack *s, int *value) {
    if (s == NULL || value == NULL || s->top < 0) {
        return -1;
    }
    *value = s->items[s->top--];
    return 0;
}

int stack_is_empty(const Stack *s) {
    return (s == NULL || s->top < 0);
}

int main(void) {
    Stack mystack;
    stack_init(&mystack);
    
    stack_push(&mystack, 10);
    stack_push(&mystack, 20);
    stack_push(&mystack, 30);
    
    int val;
    while (!stack_is_empty(&mystack)) {
        if (stack_pop(&mystack, &val) == 0) {
            printf("Popped: %d\n", val);
        }
    }
    
    return 0;
}
