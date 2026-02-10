#ifndef LLM_INTERFACE_H
#define LLM_INTERFACE_H

#include "esp_err.h"
#include "cJSON.h"
#include <stdbool.h>
#include "mimi_config.h"

typedef struct {
    char id[64];
    char name[64];
    char *input;     /* JSON string, ownership transferred to caller */
    size_t input_len;
} llm_tool_call_t;

typedef struct {
    char *text;      /* Heap-allocated, null-terminated */
    size_t text_len;
    llm_tool_call_t calls[MIMI_MAX_TOOL_CALLS];
    int call_count;
    bool tool_use;
} llm_response_t;

typedef struct {
    const char *name;
    esp_err_t (*init)(void);
    esp_err_t (*chat_tools)(const char *system_prompt,
                           cJSON *messages,
                           const char *tools_json,
                           llm_response_t *resp);
} llm_provider_t;

#endif /* LLM_INTERFACE_H */
