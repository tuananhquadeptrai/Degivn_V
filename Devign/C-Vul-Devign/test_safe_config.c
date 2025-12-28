/* Sample from Devign Dataset - FFmpeg Configuration Parser */
/* Label: SAFE (target=0) - Proper input validation */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_CONFIG_SIZE 1024
#define CONFIG_VERSION 1

typedef struct {
    int id;
    char name[64];
    int value;
    int flags;
} ConfigEntry;

typedef struct {
    int version;
    int entry_count;
    ConfigEntry *entries;
} ConfigContext;

static int validate_config_size(int size) {
    if (size < 0 || size > MAX_CONFIG_SIZE) {
        return -1;
    }
    return 0;
}

static int parse_config_entry(const char *data, int size, ConfigEntry *entry) {
    if (data == NULL || entry == NULL) {
        return -1;
    }
    
    if (size < sizeof(int) * 2) {
        return -1;
    }
    
    entry->id = *(int*)data;
    entry->value = *(int*)(data + sizeof(int));
    entry->flags = 0;
    
    return 0;
}

int load_config(ConfigContext *ctx, const char *data, int size) {
    int i;
    int offset = 0;
    
    if (ctx == NULL || data == NULL) {
        fprintf(stderr, "Invalid parameters\n");
        return -1;
    }
    
    if (validate_config_size(size) < 0) {
        fprintf(stderr, "Invalid config size: %d\n", size);
        return -1;
    }
    
    if (size < sizeof(int) * 2) {
        fprintf(stderr, "Config too small\n");
        return -1;
    }
    
    ctx->version = *(int*)data;
    offset += sizeof(int);
    
    if (ctx->version != CONFIG_VERSION) {
        fprintf(stderr, "Unsupported version: %d\n", ctx->version);
        return -1;
    }
    
    ctx->entry_count = *(int*)(data + offset);
    offset += sizeof(int);
    
    if (ctx->entry_count < 0 || ctx->entry_count > 100) {
        fprintf(stderr, "Invalid entry count: %d\n", ctx->entry_count);
        return -1;
    }
    
    if (ctx->entry_count == 0) {
        ctx->entries = NULL;
        return 0;
    }
    
    ctx->entries = calloc(ctx->entry_count, sizeof(ConfigEntry));
    if (ctx->entries == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    
    for (i = 0; i < ctx->entry_count; i++) {
        int remaining = size - offset;
        if (remaining < sizeof(int) * 2) {
            fprintf(stderr, "Truncated config at entry %d\n", i);
            free(ctx->entries);
            ctx->entries = NULL;
            return -1;
        }
        
        if (parse_config_entry(data + offset, remaining, &ctx->entries[i]) < 0) {
            free(ctx->entries);
            ctx->entries = NULL;
            return -1;
        }
        
        offset += sizeof(int) * 2;
    }
    
    printf("Loaded %d config entries\n", ctx->entry_count);
    return 0;
}

void free_config(ConfigContext *ctx) {
    if (ctx != NULL && ctx->entries != NULL) {
        free(ctx->entries);
        ctx->entries = NULL;
        ctx->entry_count = 0;
    }
}

int main(void) {
    ConfigContext ctx = {0};
    char test_data[32];
    
    memset(test_data, 0, sizeof(test_data));
    *(int*)test_data = CONFIG_VERSION;
    *(int*)(test_data + 4) = 2;
    *(int*)(test_data + 8) = 1;
    *(int*)(test_data + 12) = 100;
    *(int*)(test_data + 16) = 2;
    *(int*)(test_data + 20) = 200;
    
    if (load_config(&ctx, test_data, sizeof(test_data)) == 0) {
        printf("Config loaded successfully\n");
        printf("Version: %d, Entries: %d\n", ctx.version, ctx.entry_count);
    }
    
    free_config(&ctx);
    return 0;
}
