#include "openai.h"
#include "mimi_config.h"
#include "proxy/http_proxy.h"
#include "llm/llm_proxy.h"

#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "nvs.h"
#include "cJSON.h"

static const char *TAG = "openai";

static char s_api_key[128] = {0};
static char s_model[64] = "gpt-4o-mini"; /* Default for OpenAI */

/* ── Response buffer (duplicated for now, matching other providers) ── */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} resp_buf_t;

static esp_err_t resp_buf_init(resp_buf_t *rb, size_t initial_cap)
{
    rb->data = heap_caps_calloc(1, initial_cap, MALLOC_CAP_SPIRAM);
    if (!rb->data) return ESP_ERR_NO_MEM;
    rb->len = 0;
    rb->cap = initial_cap;
    return ESP_OK;
}

static esp_err_t resp_buf_append(resp_buf_t *rb, const char *data, size_t len)
{
    while (rb->len + len >= rb->cap) {
        size_t new_cap = rb->cap * 2;
        char *tmp = heap_caps_realloc(rb->data, new_cap, MALLOC_CAP_SPIRAM);
        if (!tmp) return ESP_ERR_NO_MEM;
        rb->data = tmp;
        rb->cap = new_cap;
    }
    memcpy(rb->data + rb->len, data, len);
    rb->len += len;
    rb->data[rb->len] = '\0';
    return ESP_OK;
}

static void resp_buf_free(resp_buf_t *rb)
{
    free(rb->data);
    rb->data = NULL;
    rb->len = 0;
    rb->cap = 0;
}

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    resp_buf_t *rb = (resp_buf_t *)evt->user_data;
    if (evt->event_id == HTTP_EVENT_ON_DATA) {
        resp_buf_append(rb, (const char *)evt->data, evt->data_len);
    }
    return ESP_OK;
}

/* ── HTTP Call Helpers ───────────────────────────────────────── */

static esp_err_t openai_http_direct(const char *post_data, resp_buf_t *rb, int *out_status)
{
    esp_http_client_config_t config = {
        .url = MIMI_OPENAI_API_URL,
        .event_handler = http_event_handler,
        .user_data = rb,
        .timeout_ms = 120 * 1000,
        .buffer_size = 4096,
        .buffer_size_tx = 4096,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) return ESP_FAIL;

    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    char auth[160];
    snprintf(auth, sizeof(auth), "Bearer %s", s_api_key);
    esp_http_client_set_header(client, "Authorization", auth);
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_perform(client);
    *out_status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    return err;
}

static esp_err_t openai_http_via_proxy(const char *post_data, resp_buf_t *rb, int *out_status)
{
    proxy_conn_t *conn = proxy_conn_open("api.openai.com", 443, 30000);
    if (!conn) return ESP_ERR_HTTP_CONNECT;

    int body_len = strlen(post_data);
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "POST /v1/chat/completions HTTP/1.1
"
        "Host: api.openai.com
"
        "Authorization: Bearer %s
"
        "Content-Type: application/json
"
        "Content-Length: %d
"
        "Connection: close

",
        s_api_key, body_len);

    if (proxy_conn_write(conn, header, hlen) < 0 ||
        proxy_conn_write(conn, post_data, body_len) < 0) {
        proxy_conn_close(conn);
        return ESP_ERR_HTTP_WRITE_DATA;
    }

    char tmp[4096];
    while (1) {
        int n = proxy_conn_read(conn, tmp, sizeof(tmp), 120000);
        if (n <= 0) break;
        if (resp_buf_append(rb, tmp, n) != ESP_OK) break;
    }
    proxy_conn_close(conn);

    *out_status = 0;
    if (rb->len > 5 && strncmp(rb->data, "HTTP/", 5) == 0) {
        const char *sp = strchr(rb->data, ' ');
        if (sp) *out_status = atoi(sp + 1);
    }

    char *body = strstr(rb->data, "

");
    if (body) {
        body += 4;
        size_t blen = rb->len - (body - rb->data);
        memmove(rb->data, body, blen);
        rb->len = blen;
        rb->data[rb->len] = '\0';
    }

    return ESP_OK;
}

static esp_err_t openai_http_call(const char *post_data, resp_buf_t *rb, int *out_status)
{
    if (http_proxy_is_enabled()) {
        return openai_http_via_proxy(post_data, rb, out_status);
    } else {
        return openai_http_direct(post_data, rb, out_status);
    }
}

/* ── Format Conversion ────────────────────────────────────────── */

static cJSON *convert_tools_openai(const char *tools_json)
{
    if (!tools_json) return NULL;
    cJSON *arr = cJSON_Parse(tools_json);
    if (!arr || !cJSON_IsArray(arr)) {
        cJSON_Delete(arr);
        return NULL;
    }
    cJSON *out = cJSON_CreateArray();
    cJSON *tool;
    cJSON_ArrayForEach(tool, arr) {
        cJSON *name = cJSON_GetObjectItem(tool, "name");
        cJSON *desc = cJSON_GetObjectItem(tool, "description");
        cJSON *schema = cJSON_GetObjectItem(tool, "input_schema");
        if (!name || !cJSON_IsString(name)) continue;

        cJSON *func = cJSON_CreateObject();
        cJSON_AddStringToObject(func, "name", name->valuestring);
        if (desc && cJSON_IsString(desc)) {
            cJSON_AddStringToObject(func, "description", desc->valuestring);
        }
        if (schema) {
            cJSON_AddItemToObject(func, "parameters", cJSON_Duplicate(schema, 1));
        }

        cJSON *wrap = cJSON_CreateObject();
        cJSON_AddStringToObject(wrap, "type", "function");
        cJSON_AddItemToObject(wrap, "function", func);
        cJSON_AddItemToArray(out, wrap);
    }
    cJSON_Delete(arr);
    return out;
}

static cJSON *convert_messages_openai(const char *system_prompt, cJSON *messages)
{
    cJSON *out = cJSON_CreateArray();
    if (system_prompt && system_prompt[0]) {
        cJSON *sys = cJSON_CreateObject();
        cJSON_AddStringToObject(sys, "role", "system");
        cJSON_AddStringToObject(sys, "content", system_prompt);
        cJSON_AddItemToArray(out, sys);
    }

    if (!messages || !cJSON_IsArray(messages)) return out;

    cJSON *msg;
    cJSON_ArrayForEach(msg, messages) {
        cJSON *role = cJSON_GetObjectItem(msg, "role");
        cJSON *content = cJSON_GetObjectItem(msg, "content");
        if (!role || !cJSON_IsString(role)) continue;

        if (cJSON_IsString(content)) {
            cJSON *m = cJSON_CreateObject();
            cJSON_AddStringToObject(m, "role", role->valuestring);
            cJSON_AddStringToObject(m, "content", content->valuestring);
            cJSON_AddItemToArray(out, m);
        } else if (cJSON_IsArray(content)) {
            /* Handle tool use/result in content array (Anthropic style internally) */
            if (strcmp(role->valuestring, "assistant") == 0) {
                cJSON *m = cJSON_CreateObject();
                cJSON_AddStringToObject(m, "role", "assistant");
                
                cJSON *tool_calls = cJSON_CreateArray();
                char *text_content = NULL;
                
                cJSON *block;
                cJSON_ArrayForEach(block, content) {
                    cJSON *type = cJSON_GetObjectItem(block, "type");
                    if (strcmp(type->valuestring, "text") == 0) {
                        text_content = cJSON_GetObjectItem(block, "text")->valuestring;
                    } else if (strcmp(type->valuestring, "tool_use") == 0) {
                        cJSON *tc = cJSON_CreateObject();
                        cJSON_AddStringToObject(tc, "id", cJSON_GetObjectItem(block, "id")->valuestring);
                        cJSON_AddStringToObject(tc, "type", "function");
                        cJSON *func = cJSON_CreateObject();
                        cJSON_AddStringToObject(func, "name", cJSON_GetObjectItem(block, "name")->valuestring);
                        cJSON_AddItemToObject(func, "arguments", cJSON_PrintUnformatted(cJSON_GetObjectItem(block, "input")));
                        cJSON_AddItemToObject(tc, "function", func);
                        cJSON_AddItemToArray(tool_calls, tc);
                    }
                }
                
                if (text_content) cJSON_AddStringToObject(m, "content", text_content);
                if (cJSON_GetArraySize(tool_calls) > 0) {
                    cJSON_AddItemToObject(m, "tool_calls", tool_calls);
                } else {
                    cJSON_Delete(tool_calls);
                }
                cJSON_AddItemToArray(out, m);
            } else if (strcmp(role->valuestring, "user") == 0) {
                /* User content array (tool results) */
                cJSON *block;
                cJSON_ArrayForEach(block, content) {
                    cJSON *type = cJSON_GetObjectItem(block, "type");
                    if (strcmp(type->valuestring, "tool_result") == 0) {
                        cJSON *m = cJSON_CreateObject();
                        cJSON_AddStringToObject(m, "role", "tool");
                        cJSON_AddStringToObject(m, "tool_call_id", cJSON_GetObjectItem(block, "tool_use_id")->valuestring);
                        cJSON_AddStringToObject(m, "content", cJSON_GetObjectItem(block, "content")->valuestring);
                        cJSON_AddItemToArray(out, m);
                    } else if (strcmp(type->valuestring, "text") == 0) {
                        cJSON *m = cJSON_CreateObject();
                        cJSON_AddStringToObject(m, "role", "user");
                        cJSON_AddStringToObject(m, "content", cJSON_GetObjectItem(block, "text")->valuestring);
                        cJSON_AddItemToArray(out, m);
                    }
                }
            }
        }
    }
    return out;
}

/* ── Provider Implementation ──────────────────────────────────── */

static esp_err_t openai_init(void)
{
    /* Start with build-time defaults */
#ifdef MIMI_SECRET_API_KEY_OAI
    if (MIMI_SECRET_API_KEY_OAI[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY_OAI, sizeof(s_api_key) - 1);
    }
#endif
    if (s_api_key[0] == '\0' && MIMI_SECRET_API_KEY[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY, sizeof(s_api_key) - 1);
    }

    nvs_handle_t nvs;
    if (nvs_open(MIMI_NVS_LLM, NVS_READONLY, &nvs) == ESP_OK) {
        char tmp[128] = {0};
        size_t len = sizeof(tmp);
        if (nvs_get_str(nvs, "api_key_oai", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_api_key, tmp, sizeof(s_api_key) - 1);
        } else {
            /* Fallback to generic api_key if specifically for openai not set */
            len = sizeof(tmp);
            if (nvs_get_str(nvs, MIMI_NVS_KEY_API_KEY, tmp, &len) == ESP_OK && tmp[0]) {
                 strncpy(s_api_key, tmp, sizeof(s_api_key) - 1);
            }
        }
        
        len = sizeof(tmp);
        if (nvs_get_str(nvs, "model_oai", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_model, tmp, sizeof(s_model) - 1);
        }
        nvs_close(nvs);
    }
    
    if (s_api_key[0] == '\0') {
        ESP_LOGW(TAG, "No OpenAI API key. Use CLI: set_api_key <KEY>");
    }
    ESP_LOGI(TAG, "OpenAI provider initialized (model: %s)", s_model);
    return ESP_OK;
}

static esp_err_t openai_chat_tools(const char *system_prompt,
                                   cJSON *messages,
                                   const char *tools_json,
                                   llm_response_t *resp)
{
    memset(resp, 0, sizeof(*resp));
    if (s_api_key[0] == '\0') return ESP_ERR_INVALID_STATE;

    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", s_model);
    cJSON_AddItemToObject(body, "messages", convert_messages_openai(system_prompt, messages));

    if (tools_json) {
        cJSON *tools = convert_tools_openai(tools_json);
        if (tools) {
            cJSON_AddItemToObject(body, "tools", tools);
            cJSON_AddStringToObject(body, "tool_choice", "auto");
        }
    }

    char *post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) return ESP_ERR_NO_MEM;

    resp_buf_t rb;
    resp_buf_init(&rb, MIMI_LLM_STREAM_BUF_SIZE);

    int status = 0;
    esp_err_t err = openai_http_call(post_data, &rb, &status);
    free(post_data);

    if (err != ESP_OK || status != 200) {
        ESP_LOGE(TAG, "API error %d: %.500s", status, rb.data ? rb.data : "");
        resp_buf_free(&rb);
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(rb.data);
    resp_buf_free(&rb);
    if (!root) return ESP_FAIL;

    cJSON *choices = cJSON_GetObjectItem(root, "choices");
    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
        cJSON *choice = cJSON_GetArrayItem(choices, 0);
        cJSON *message = cJSON_GetObjectItem(choice, "message");
        
        if (message) {
            cJSON *content = cJSON_GetObjectItem(message, "content");
            if (content && cJSON_IsString(content)) {
                resp->text = strdup(content->valuestring);
                resp->text_len = strlen(resp->text);
            }
            
            cJSON *tool_calls = cJSON_GetObjectItem(message, "tool_calls");
            if (tool_calls && cJSON_IsArray(tool_calls)) {
                int n = cJSON_GetArraySize(tool_calls);
                for (int i = 0; i < n && resp->call_count < MIMI_MAX_TOOL_CALLS; i++) {
                    cJSON *tc = cJSON_GetArrayItem(tool_calls, i);
                    cJSON *func = cJSON_GetObjectItem(tc, "function");
                    if (func) {
                        llm_tool_call_t *call = &resp->calls[resp->call_count];
                        strncpy(call->id, cJSON_GetObjectItem(tc, "id")->valuestring, sizeof(call->id) - 1);
                        strncpy(call->name, cJSON_GetObjectItem(func, "name")->valuestring, sizeof(call->name) - 1);
                        call->input = strdup(cJSON_GetObjectItem(func, "arguments")->valuestring);
                        call->input_len = strlen(call->input);
                        resp->call_count++;
                        resp->tool_use = true;
                    }
                }
            }
        }
    }

    cJSON_Delete(root);
    return ESP_OK;
}

const llm_provider_t openai_provider = {
    .name = "openai",
    .init = openai_init,
    .chat_tools = openai_chat_tools,
};
