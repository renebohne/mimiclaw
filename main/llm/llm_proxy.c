#include "llm_proxy.h"
#include "llm_interface.h"
#include "providers/anthropic.h"
#include "providers/gemini.h"
#include "providers/openai.h"
#include "mimi_config.h"

#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "nvs.h"

static const char *TAG = "llm_mgr";

static const llm_provider_t *s_providers[] = {
    &anthropic_provider,
    &gemini_provider,
    &openai_provider,
    NULL
};

static const llm_provider_t *s_active_provider = NULL;

esp_err_t llm_proxy_init(void)
{
    char provider_name[32] = "gemini"; /* Default */

    /* Load provider choice from NVS */
    nvs_handle_t nvs;
    if (nvs_open(MIMI_NVS_LLM, NVS_READONLY, &nvs) == ESP_OK) {
        size_t len = sizeof(provider_name);
        nvs_get_str(nvs, MIMI_NVS_KEY_PROVIDER, provider_name, &len);
        nvs_close(nvs);
    }

    /* Find provider */
    for (int i = 0; s_providers[i] != NULL; i++) {
        if (strcmp(s_providers[i]->name, provider_name) == 0) {
            s_active_provider = s_providers[i];
            break;
        }
    }

    if (!s_active_provider) {
        ESP_LOGW(TAG, "Provider '%s' not found, falling back to anthropic", provider_name);
        s_active_provider = &anthropic_provider;
    }

    ESP_LOGI(TAG, "Active provider: %s", s_active_provider->name);
    return s_active_provider->init();
}

esp_err_t llm_chat_tools(const char *system_prompt,
                         cJSON *messages,
                         const char *tools_json,
                         llm_response_t *resp)
{
    memset(resp, 0, sizeof(*resp));
    if (!s_active_provider) return ESP_ERR_INVALID_STATE;
    return s_active_provider->chat_tools(system_prompt, messages, tools_json, resp);
}

void llm_response_free(llm_response_t *resp)
{
    free(resp->text);
    resp->text = NULL;
    resp->text_len = 0;
    for (int i = 0; i < resp->call_count; i++) {
        free(resp->calls[i].input);
        resp->calls[i].input = NULL;
    }
    resp->call_count = 0;
    resp->tool_use = false;
}

/* Backward compatibility for simple chat */
esp_err_t llm_chat(const char *system_prompt, const char *messages_json,
                   char *response_buf, size_t buf_size)
{
    cJSON *messages = cJSON_Parse(messages_json);
    if (!messages) return ESP_ERR_INVALID_ARG;

    llm_response_t resp;
    esp_err_t err = llm_chat_tools(system_prompt, messages, NULL, &resp);
    cJSON_Delete(messages);

    if (err == ESP_OK) {
        if (resp.text) {
            strncpy(response_buf, resp.text, buf_size - 1);
            response_buf[buf_size - 1] = '\0';
        } else {
            response_buf[0] = '\0';
        }
        llm_response_free(&resp);
    }
    return err;
}

/* ── NVS helpers ──────────────────────────────────────────────── */

esp_err_t llm_set_provider(const char *name)
{
    nvs_handle_t nvs;
    esp_err_t err = nvs_open(MIMI_NVS_LLM, NVS_READWRITE, &nvs);
    if (err != ESP_OK) return err;
    
    err = nvs_set_str(nvs, MIMI_NVS_KEY_PROVIDER, name);
    if (err == ESP_OK) err = nvs_commit(nvs);
    nvs_close(nvs);

    ESP_LOGI(TAG, "Provider set to: %s (restart required)", name);
    return err;
}

esp_err_t llm_set_api_key(const char *api_key)
{
    if (!s_active_provider) return ESP_ERR_INVALID_STATE;
    
    char key_name[32];
    snprintf(key_name, sizeof(key_name), "api_key_%s", s_active_provider->name);
    if (strlen(s_active_provider->name) > 3) {
        /* shortcut for known providers to stay compatible with existing naming if needed */
        if (strcmp(s_active_provider->name, "anthropic") == 0) strcpy(key_name, "api_key_ant");
        if (strcmp(s_active_provider->name, "gemini") == 0) strcpy(key_name, "api_key_gem");
        if (strcmp(s_active_provider->name, "openai") == 0) strcpy(key_name, "api_key_oai");
    }

    nvs_handle_t nvs;
    ESP_ERROR_CHECK(nvs_open(MIMI_NVS_LLM, NVS_READWRITE, &nvs));
    ESP_ERROR_CHECK(nvs_set_str(nvs, key_name, api_key));
    ESP_ERROR_CHECK(nvs_commit(nvs));
    nvs_close(nvs);

    ESP_LOGI(TAG, "API key saved for provider %s", s_active_provider->name);
    return ESP_OK;
}

esp_err_t llm_set_model(const char *model)
{
    if (!s_active_provider) return ESP_ERR_INVALID_STATE;

    char key_name[32];
    snprintf(key_name, sizeof(key_name), "model_%s", s_active_provider->name);
    if (strcmp(s_active_provider->name, "anthropic") == 0) strcpy(key_name, "model_ant");
    if (strcmp(s_active_provider->name, "gemini") == 0) strcpy(key_name, "model_gem");
    if (strcmp(s_active_provider->name, "openai") == 0) strcpy(key_name, "model_oai");

    nvs_handle_t nvs;
    ESP_ERROR_CHECK(nvs_open(MIMI_NVS_LLM, NVS_READWRITE, &nvs));
    ESP_ERROR_CHECK(nvs_set_str(nvs, key_name, model));
    ESP_ERROR_CHECK(nvs_commit(nvs));
    nvs_close(nvs);

    ESP_LOGI(TAG, "Model saved for provider %s", s_active_provider->name);
    return ESP_OK;
}
