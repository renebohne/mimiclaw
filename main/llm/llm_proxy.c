#include "llm_proxy.h"
#include "mimi_config.h"
#include "proxy/http_proxy.h"

#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "nvs.h"
#include "cJSON.h"

static const char *TAG = "llm";

static char s_api_key[128] = {0};
static char s_model[64] = MIMI_LLM_DEFAULT_MODEL;

/* ── Response buffer ──────────────────────────────────────────── */

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

/* ── HTTP event handler (for esp_http_client direct path) ─────── */

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    resp_buf_t *rb = (resp_buf_t *)evt->user_data;
    if (evt->event_id == HTTP_EVENT_ON_DATA) {
        resp_buf_append(rb, (const char *)evt->data, evt->data_len);
    }
    return ESP_OK;
}

/* ── Init ─────────────────────────────────────────────────────── */

esp_err_t llm_proxy_init(void)
{
    /* Build-time secrets take highest priority */
    if (MIMI_SECRET_API_KEY[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY, sizeof(s_api_key) - 1);
    }
    if (MIMI_SECRET_MODEL[0] != '\0') {
        strncpy(s_model, MIMI_SECRET_MODEL, sizeof(s_model) - 1);
    }

    /* Fall back to NVS for values not set at build time */
    if (s_api_key[0] == '\0' || s_model[0] == '\0') {
        nvs_handle_t nvs;
        esp_err_t err = nvs_open(MIMI_NVS_LLM, NVS_READONLY, &nvs);
        if (err == ESP_OK) {
            if (s_api_key[0] == '\0') {
                size_t len = sizeof(s_api_key);
                nvs_get_str(nvs, MIMI_NVS_KEY_API_KEY, s_api_key, &len);
            }
            if (strcmp(s_model, MIMI_LLM_DEFAULT_MODEL) == 0) {
                size_t len = sizeof(s_model);
                nvs_get_str(nvs, MIMI_NVS_KEY_MODEL, s_model, &len);
            }
            nvs_close(nvs);
        }
    }

    if (s_api_key[0]) {
        ESP_LOGI(TAG, "LLM proxy initialized (model: %s)", s_model);
    } else {
        ESP_LOGW(TAG, "No API key. Use CLI: set_api_key <KEY>");
    }
    return ESP_OK;
}

/* ── Direct path: esp_http_client ───────────────────────────── */

static esp_err_t llm_http_direct(const char *post_data, resp_buf_t *rb, int *out_status)
{
    esp_http_client_config_t config = {
        .url = MIMI_LLM_API_URL,
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
    esp_http_client_set_header(client, "x-api-key", s_api_key);
    esp_http_client_set_header(client, "anthropic-version", MIMI_LLM_API_VERSION);
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_perform(client);
    *out_status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    return err;
}

/* ── Proxy path: manual HTTP over CONNECT tunnel ────────────── */

static esp_err_t llm_http_via_proxy(const char *post_data, resp_buf_t *rb, int *out_status)
{
    proxy_conn_t *conn = proxy_conn_open("api.anthropic.com", 443, 30000);
    if (!conn) return ESP_ERR_HTTP_CONNECT;

    int body_len = strlen(post_data);
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "POST /v1/messages HTTP/1.1\r\n"
        "Host: api.anthropic.com\r\n"
        "Content-Type: application/json\r\n"
        "x-api-key: %s\r\n"
        "anthropic-version: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n",
        s_api_key, MIMI_LLM_API_VERSION, body_len);

    if (proxy_conn_write(conn, header, hlen) < 0 ||
        proxy_conn_write(conn, post_data, body_len) < 0) {
        proxy_conn_close(conn);
        return ESP_ERR_HTTP_WRITE_DATA;
    }

    /* Read full response into buffer */
    char tmp[4096];
    while (1) {
        int n = proxy_conn_read(conn, tmp, sizeof(tmp), 120000);
        if (n <= 0) break;
        if (resp_buf_append(rb, tmp, n) != ESP_OK) break;
    }
    proxy_conn_close(conn);

    /* Parse status line */
    *out_status = 0;
    if (rb->len > 5 && strncmp(rb->data, "HTTP/", 5) == 0) {
        const char *sp = strchr(rb->data, ' ');
        if (sp) *out_status = atoi(sp + 1);
    }

    /* Strip HTTP headers, keep body only */
    char *body = strstr(rb->data, "\r\n\r\n");
    if (body) {
        body += 4;
        size_t blen = rb->len - (body - rb->data);
        memmove(rb->data, body, blen);
        rb->len = blen;
        rb->data[rb->len] = '\0';
    }

    return ESP_OK;
}

/* ── Shared HTTP dispatch ─────────────────────────────────────── */

static esp_err_t llm_http_call(const char *post_data, resp_buf_t *rb, int *out_status)
{
    if (http_proxy_is_enabled()) {
        return llm_http_via_proxy(post_data, rb, out_status);
    } else {
        return llm_http_direct(post_data, rb, out_status);
    }
}

/* ── Parse text from JSON response ────────────────────────────── */

static void extract_text(cJSON *root, char *buf, size_t size)
{
    buf[0] = '\0';
    cJSON *content = cJSON_GetObjectItem(root, "content");
    if (!content || !cJSON_IsArray(content)) return;

    size_t off = 0;
    cJSON *block;
    cJSON_ArrayForEach(block, content) {
        cJSON *btype = cJSON_GetObjectItem(block, "type");
        if (!btype || strcmp(btype->valuestring, "text") != 0) continue;
        cJSON *text = cJSON_GetObjectItem(block, "text");
        if (!text || !cJSON_IsString(text)) continue;
        size_t tlen = strlen(text->valuestring);
        size_t copy = (tlen < size - off - 1) ? tlen : size - off - 1;
        memcpy(buf + off, text->valuestring, copy);
        off += copy;
    }
    buf[off] = '\0';
}

/* ── Public: simple chat (backward compat) ────────────────────── */

esp_err_t llm_chat(const char *system_prompt, const char *messages_json,
                   char *response_buf, size_t buf_size)
{
    if (s_api_key[0] == '\0') {
        snprintf(response_buf, buf_size, "Error: No API key configured");
        return ESP_ERR_INVALID_STATE;
    }

    /* Build request body (non-streaming) */
    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", s_model);
    cJSON_AddNumberToObject(body, "max_tokens", MIMI_LLM_MAX_TOKENS);
    cJSON_AddStringToObject(body, "system", system_prompt);

    cJSON *messages = cJSON_Parse(messages_json);
    if (messages) {
        cJSON_AddItemToObject(body, "messages", messages);
    } else {
        cJSON *arr = cJSON_CreateArray();
        cJSON *msg = cJSON_CreateObject();
        cJSON_AddStringToObject(msg, "role", "user");
        cJSON_AddStringToObject(msg, "content", messages_json);
        cJSON_AddItemToArray(arr, msg);
        cJSON_AddItemToObject(body, "messages", arr);
    }

    char *post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) {
        snprintf(response_buf, buf_size, "Error: Failed to build request");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "Calling Claude API (model: %s, body: %d bytes)",
             s_model, (int)strlen(post_data));

    resp_buf_t rb;
    if (resp_buf_init(&rb, MIMI_LLM_STREAM_BUF_SIZE) != ESP_OK) {
        free(post_data);
        snprintf(response_buf, buf_size, "Error: Out of memory");
        return ESP_ERR_NO_MEM;
    }

    int status = 0;
    esp_err_t err = llm_http_call(post_data, &rb, &status);
    free(post_data);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP request failed: %s", esp_err_to_name(err));
        resp_buf_free(&rb);
        snprintf(response_buf, buf_size, "Error: HTTP request failed (%s)",
                 esp_err_to_name(err));
        return err;
    }

    if (status != 200) {
        ESP_LOGE(TAG, "API returned status %d", status);
        snprintf(response_buf, buf_size, "API error (HTTP %d): %.200s",
                 status, rb.data ? rb.data : "");
        resp_buf_free(&rb);
        return ESP_FAIL;
    }

    /* Parse JSON response */
    cJSON *root = cJSON_Parse(rb.data);
    resp_buf_free(&rb);

    if (!root) {
        snprintf(response_buf, buf_size, "Error: Failed to parse response");
        return ESP_FAIL;
    }

    extract_text(root, response_buf, buf_size);
    cJSON_Delete(root);

    if (response_buf[0] == '\0') {
        snprintf(response_buf, buf_size, "No response from Claude API");
    } else {
        ESP_LOGI(TAG, "Claude response: %d bytes", (int)strlen(response_buf));
    }

    return ESP_OK;
}

/* ── Public: chat with tools (non-streaming) ──────────────────── */

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

esp_err_t llm_chat_tools(const char *system_prompt,
                         cJSON *messages,
                         const char *tools_json,
                         llm_response_t *resp)
{
    memset(resp, 0, sizeof(*resp));

    if (s_api_key[0] == '\0') return ESP_ERR_INVALID_STATE;

    /* Build request body (non-streaming) */
    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", s_model);
    cJSON_AddNumberToObject(body, "max_tokens", MIMI_LLM_MAX_TOKENS);
    cJSON_AddStringToObject(body, "system", system_prompt);

    /* Deep-copy messages so caller keeps ownership */
    cJSON *msgs_copy = cJSON_Duplicate(messages, 1);
    cJSON_AddItemToObject(body, "messages", msgs_copy);

    /* Add tools array if provided */
    if (tools_json) {
        cJSON *tools = cJSON_Parse(tools_json);
        if (tools) {
            cJSON_AddItemToObject(body, "tools", tools);
        }
    }

    char *post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) return ESP_ERR_NO_MEM;

    ESP_LOGI(TAG, "Calling Claude API with tools (model: %s, body: %d bytes)",
             s_model, (int)strlen(post_data));

    /* HTTP call */
    resp_buf_t rb;
    if (resp_buf_init(&rb, MIMI_LLM_STREAM_BUF_SIZE) != ESP_OK) {
        free(post_data);
        return ESP_ERR_NO_MEM;
    }

    int status = 0;
    esp_err_t err = llm_http_call(post_data, &rb, &status);
    free(post_data);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP request failed: %s", esp_err_to_name(err));
        resp_buf_free(&rb);
        return err;
    }

    if (status != 200) {
        ESP_LOGE(TAG, "API error %d: %.500s", status, rb.data ? rb.data : "");
        resp_buf_free(&rb);
        return ESP_FAIL;
    }

    /* Parse full JSON response */
    cJSON *root = cJSON_Parse(rb.data);
    resp_buf_free(&rb);

    if (!root) {
        ESP_LOGE(TAG, "Failed to parse API response JSON");
        return ESP_FAIL;
    }

    /* stop_reason */
    cJSON *stop_reason = cJSON_GetObjectItem(root, "stop_reason");
    if (stop_reason && cJSON_IsString(stop_reason)) {
        resp->tool_use = (strcmp(stop_reason->valuestring, "tool_use") == 0);
    }

    /* Iterate content blocks */
    cJSON *content = cJSON_GetObjectItem(root, "content");
    if (content && cJSON_IsArray(content)) {
        /* Accumulate total text length first */
        size_t total_text = 0;
        cJSON *block;
        cJSON_ArrayForEach(block, content) {
            cJSON *btype = cJSON_GetObjectItem(block, "type");
            if (btype && strcmp(btype->valuestring, "text") == 0) {
                cJSON *text = cJSON_GetObjectItem(block, "text");
                if (text && cJSON_IsString(text)) {
                    total_text += strlen(text->valuestring);
                }
            }
        }

        /* Allocate and copy text */
        if (total_text > 0) {
            resp->text = calloc(1, total_text + 1);
            if (resp->text) {
                cJSON_ArrayForEach(block, content) {
                    cJSON *btype = cJSON_GetObjectItem(block, "type");
                    if (!btype || strcmp(btype->valuestring, "text") != 0) continue;
                    cJSON *text = cJSON_GetObjectItem(block, "text");
                    if (!text || !cJSON_IsString(text)) continue;
                    size_t tlen = strlen(text->valuestring);
                    memcpy(resp->text + resp->text_len, text->valuestring, tlen);
                    resp->text_len += tlen;
                }
                resp->text[resp->text_len] = '\0';
            }
        }

        /* Extract tool_use blocks */
        cJSON_ArrayForEach(block, content) {
            cJSON *btype = cJSON_GetObjectItem(block, "type");
            if (!btype || strcmp(btype->valuestring, "tool_use") != 0) continue;
            if (resp->call_count >= MIMI_MAX_TOOL_CALLS) break;

            llm_tool_call_t *call = &resp->calls[resp->call_count];

            cJSON *id = cJSON_GetObjectItem(block, "id");
            if (id && cJSON_IsString(id)) {
                strncpy(call->id, id->valuestring, sizeof(call->id) - 1);
            }

            cJSON *name = cJSON_GetObjectItem(block, "name");
            if (name && cJSON_IsString(name)) {
                strncpy(call->name, name->valuestring, sizeof(call->name) - 1);
            }

            cJSON *input = cJSON_GetObjectItem(block, "input");
            if (input) {
                char *input_str = cJSON_PrintUnformatted(input);
                if (input_str) {
                    call->input = input_str;
                    call->input_len = strlen(input_str);
                }
            }

            resp->call_count++;
        }
    }

    cJSON_Delete(root);

    ESP_LOGI(TAG, "Response: %d bytes text, %d tool calls, stop=%s",
             (int)resp->text_len, resp->call_count,
             resp->tool_use ? "tool_use" : "end_turn");

    return ESP_OK;
}

/* ── NVS helpers ──────────────────────────────────────────────── */

esp_err_t llm_set_api_key(const char *api_key)
{
    nvs_handle_t nvs;
    ESP_ERROR_CHECK(nvs_open(MIMI_NVS_LLM, NVS_READWRITE, &nvs));
    ESP_ERROR_CHECK(nvs_set_str(nvs, MIMI_NVS_KEY_API_KEY, api_key));
    ESP_ERROR_CHECK(nvs_commit(nvs));
    nvs_close(nvs);

    strncpy(s_api_key, api_key, sizeof(s_api_key) - 1);
    ESP_LOGI(TAG, "API key saved");
    return ESP_OK;
}

esp_err_t llm_set_model(const char *model)
{
    nvs_handle_t nvs;
    ESP_ERROR_CHECK(nvs_open(MIMI_NVS_LLM, NVS_READWRITE, &nvs));
    ESP_ERROR_CHECK(nvs_set_str(nvs, MIMI_NVS_KEY_MODEL, model));
    ESP_ERROR_CHECK(nvs_commit(nvs));
    nvs_close(nvs);

    strncpy(s_model, model, sizeof(s_model) - 1);
    ESP_LOGI(TAG, "Model set to: %s", s_model);
    return ESP_OK;
}
