#pragma once

#include "esp_err.h"
#include "cJSON.h"
#include <stddef.h>
#include <stdbool.h>

#include "mimi_config.h"
#include "llm/llm_interface.h"

/**
 * Initialize the LLM proxy manager.
 */
esp_err_t llm_proxy_init(void);

/**
 * Set the active LLM provider (anthropic, gemini, etc.)
 */
esp_err_t llm_set_provider(const char *name);

/**
 * Save the API key for the current provider to NVS.
 */
esp_err_t llm_set_api_key(const char *api_key);

/**
 * Save the model identifier to NVS for the current provider.
 */
esp_err_t llm_set_model(const char *model);

/**
 * Send a chat completion request to the configured LLM API (non-streaming).
 *
 * @param system_prompt  System prompt string
 * @param messages_json  JSON array of messages: [{"role":"user","content":"..."},...]
 * @param response_buf   Output buffer for the complete response text
 * @param buf_size       Size of response_buf
 * @return ESP_OK on success
 */
esp_err_t llm_chat(const char *system_prompt, const char *messages_json,
                   char *response_buf, size_t buf_size);

void llm_response_free(llm_response_t *resp);

/**
 * Send a chat completion request with tools to the configured LLM API (non-streaming).
 *
 * @param system_prompt  System prompt string
 * @param messages       cJSON array of messages (caller owns)
 * @param tools_json     Pre-built JSON string of tools array, or NULL for no tools
 * @param resp           Output: structured response with text and tool calls
 * @return ESP_OK on success
 */
esp_err_t llm_chat_tools(const char *system_prompt,
                         cJSON *messages,
                         const char *tools_json,
                         llm_response_t *resp);
