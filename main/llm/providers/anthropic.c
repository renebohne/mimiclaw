#include "anthropic.h"
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

static const char *TAG = "anthropic";

static char s_api_key[128] = {0};
static char s_model[64] = MIMI_LLM_DEFAULT_MODEL;

/* ── Response buffer ── */

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

/* ── HTTP Call Helpers ── */

static esp_err_t anthropic_http_direct(const char *post_data, resp_buf_t *rb, int *out_status)
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

static esp_err_t anthropic_http_via_proxy(const char *post_data, resp_buf_t *rb, int *out_status)
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

static esp_err_t anthropic_http_call(const char *post_data, resp_buf_t *rb, int *out_status)
{
    if (http_proxy_is_enabled()) {
        return anthropic_http_via_proxy(post_data, rb, out_status);
    } else {
        return anthropic_http_direct(post_data, rb, out_status);
    }
}

/* ── Provider Implementation ── */

static esp_err_t anthropic_init(void)
{
#ifdef MIMI_SECRET_API_KEY_ANT
    if (MIMI_SECRET_API_KEY_ANT[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY_ANT, sizeof(s_api_key) - 1);
    }
#endif
    if (s_api_key[0] == '\0' && MIMI_SECRET_API_KEY[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY, sizeof(s_api_key) - 1);
    }

    nvs_handle_t nvs;
    if (nvs_open(MIMI_NVS_LLM, NVS_READONLY, &nvs) == ESP_OK) {
        char tmp[128] = {0};
        size_t len = sizeof(tmp);
        if (nvs_get_str(nvs, "api_key_ant", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_api_key, tmp, sizeof(s_api_key) - 1);
        }
        
        len = sizeof(tmp);
        if (nvs_get_str(nvs, "model_ant", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_model, tmp, sizeof(s_model) - 1);
        }
        nvs_close(nvs);
    }

    ESP_LOGI(TAG, "Anthropic provider initialized (model: %s)", s_model);
    return ESP_OK;
}

static esp_err_t anthropic_chat_tools(const char *system_prompt,
                                     cJSON *messages,
                                     const char *tools_json,
                                     llm_response_t *resp)
{
    memset(resp, 0, sizeof(*resp));
    if (s_api_key[0] == '\0') return ESP_ERR_INVALID_STATE;

    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", s_model);
    cJSON_AddNumberToObject(body, "max_tokens", MIMI_LLM_MAX_TOKENS);
    cJSON_AddStringToObject(body, "system", system_prompt);

    cJSON *msgs_copy = cJSON_Duplicate(messages, 1);
    cJSON_AddItemToObject(body, "messages", msgs_copy);

    if (tools_json) {
        cJSON *tools = cJSON_Parse(tools_json);
        if (tools) {
            cJSON_AddItemToObject(body, "tools", tools);
        }
    }

    char *post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) return ESP_ERR_NO_MEM;

    resp_buf_t rb;
    if (resp_buf_init(&rb, MIMI_LLM_STREAM_BUF_SIZE) != ESP_OK) {
        free(post_data);
        return ESP_ERR_NO_MEM;
    }

    int status = 0;
    esp_err_t err = anthropic_http_call(post_data, &rb, &status);
    free(post_data);

    if (err != ESP_OK) {
        resp_buf_free(&rb);
        return err;
    }

    if (status != 200) {
        ESP_LOGE(TAG, "API error %d: %.500s", status, rb.data ? rb.data : "");
        resp_buf_free(&rb);
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(rb.data);
    resp_buf_free(&rb);
    if (!root) return ESP_FAIL;

    cJSON *stop_reason = cJSON_GetObjectItem(root, "stop_reason");
    if (stop_reason && cJSON_IsString(stop_reason)) {
        resp->tool_use = (strcmp(stop_reason->valuestring, "tool_use") == 0);
    }

    cJSON *content = cJSON_GetObjectItem(root, "content");
    if (content && cJSON_IsArray(content)) {
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

        cJSON_ArrayForEach(block, content) {
            cJSON *btype = cJSON_GetObjectItem(block, "type");
            if (!btype || strcmp(btype->valuestring, "tool_use") != 0) continue;
            if (resp->call_count >= MIMI_MAX_TOOL_CALLS) break;

            llm_tool_call_t *call = &resp->calls[resp->call_count];
            cJSON *id = cJSON_GetObjectItem(block, "id");
            if (id && cJSON_IsString(id)) strncpy(call->id, id->valuestring, sizeof(call->id) - 1);
            cJSON *name = cJSON_GetObjectItem(block, "name");
            if (name && cJSON_IsString(name)) strncpy(call->name, name->valuestring, sizeof(call->name) - 1);
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
    return ESP_OK;
}

const llm_provider_t anthropic_provider = {
    .name = "anthropic",
    .init = anthropic_init,
    .chat_tools = anthropic_chat_tools,
};