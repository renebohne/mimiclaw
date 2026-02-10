#include "gemini.h"
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

static const char *TAG = "gemini";

static char s_api_key[128] = {0};
static char s_model[64] = MIMI_LLM_DEFAULT_MODEL;

/* ── Response buffer (duplicated for now, should refactor later) ── */

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

static esp_err_t gemini_http_direct(const char *post_data, resp_buf_t *rb, int *out_status)
{
    char url[256];
    snprintf(url, sizeof(url), "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", 
             s_model, s_api_key);

    esp_http_client_config_t config = {
        .url = url,
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
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_perform(client);
    *out_status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    return err;
}

static esp_err_t gemini_http_via_proxy(const char *post_data, resp_buf_t *rb, int *out_status)
{
    proxy_conn_t *conn = proxy_conn_open("generativelanguage.googleapis.com", 443, 30000);
    if (!conn) return ESP_ERR_HTTP_CONNECT;

    int body_len = strlen(post_data);
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "POST /v1beta/models/%s:generateContent?key=%s HTTP/1.1\r\n"
        "Host: generativelanguage.googleapis.com\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n",
        s_model, s_api_key, body_len);

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

static esp_err_t gemini_http_call(const char *post_data, resp_buf_t *rb, int *out_status)
{
    if (http_proxy_is_enabled()) {
        return gemini_http_via_proxy(post_data, rb, out_status);
    } else {
        return gemini_http_direct(post_data, rb, out_status);
    }
}

/* ── Format Conversion ────────────────────────────────────────── */

static cJSON *convert_messages_to_gemini(cJSON *messages)
{
    cJSON *contents = cJSON_CreateArray();
    cJSON *msg;
    cJSON_ArrayForEach(msg, messages) {
        cJSON *g_msg = cJSON_CreateObject();
        cJSON *role = cJSON_GetObjectItem(msg, "role");
        cJSON *content = cJSON_GetObjectItem(msg, "content");

        const char *r_str = role->valuestring;
        if (strcmp(r_str, "assistant") == 0) {
            cJSON_AddStringToObject(g_msg, "role", "model");
        } else if (strcmp(r_str, "user") == 0) {
            cJSON_AddStringToObject(g_msg, "role", "user");
        } else {
            /* Gemini is strict about roles: user or model. 
               Function results are special but handle below. */
            cJSON_AddStringToObject(g_msg, "role", "user");
        }

        cJSON *parts = cJSON_CreateArray();
        if (cJSON_IsString(content)) {
            cJSON *p = cJSON_CreateObject();
            cJSON_AddStringToObject(p, "text", content->valuestring);
            cJSON_AddItemToArray(parts, p);
        } else if (cJSON_IsArray(content)) {
            cJSON *item;
            cJSON_ArrayForEach(item, content) {
                cJSON *type = cJSON_GetObjectItem(item, "type");
                if (!type) continue;
                
                if (strcmp(type->valuestring, "text") == 0) {
                    cJSON *p = cJSON_CreateObject();
                    cJSON_AddStringToObject(p, "text", cJSON_GetObjectItem(item, "text")->valuestring);
                    cJSON_AddItemToArray(parts, p);
                } else if (strcmp(type->valuestring, "tool_use") == 0) {
                    cJSON *p = cJSON_CreateObject();
                    cJSON *funcCall = cJSON_CreateObject();
                    cJSON_AddStringToObject(funcCall, "name", cJSON_GetObjectItem(item, "name")->valuestring);
                    cJSON_AddItemToObject(funcCall, "args", cJSON_Duplicate(cJSON_GetObjectItem(item, "input"), 1));
                    cJSON_AddItemToObject(p, "functionCall", funcCall);
                    cJSON_AddItemToArray(parts, p);
                } else if (strcmp(type->valuestring, "tool_result") == 0) {
                    /* Role must be "function" for Gemini functionResponse */
                    cJSON_ReplaceItemInObject(g_msg, "role", cJSON_CreateString("function"));
                    cJSON *p = cJSON_CreateObject();
                    cJSON *funcResp = cJSON_CreateObject();
                    cJSON *name = cJSON_GetObjectItem(item, "name");
                    cJSON_AddStringToObject(funcResp, "name", name ? name->valuestring : "web_search");
                    cJSON *resp_obj = cJSON_CreateObject();
                    cJSON_AddStringToObject(resp_obj, "content", cJSON_GetObjectItem(item, "content")->valuestring);
                    cJSON_AddItemToObject(funcResp, "response", resp_obj);
                    cJSON_AddItemToObject(p, "functionResponse", funcResp);
                    cJSON_AddItemToArray(parts, p);
                }
            }
        }
        cJSON_AddItemToObject(g_msg, "parts", parts);
        cJSON_AddItemToArray(contents, g_msg);
    }
    return contents;
}

/* ── Provider Implementation ──────────────────────────────────── */

static esp_err_t gemini_init(void)
{
    /* Start with build-time defaults */
#ifdef MIMI_SECRET_API_KEY_GEM
    if (MIMI_SECRET_API_KEY_GEM[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY_GEM, sizeof(s_api_key) - 1);
    }
#endif
    if (s_api_key[0] == '\0' && MIMI_SECRET_API_KEY[0] != '\0') {
        strncpy(s_api_key, MIMI_SECRET_API_KEY, sizeof(s_api_key) - 1);
    }

    nvs_handle_t nvs;
    if (nvs_open(MIMI_NVS_LLM, NVS_READONLY, &nvs) == ESP_OK) {
        char tmp[128] = {0};
        size_t len = sizeof(tmp);
        if (nvs_get_str(nvs, "api_key_gem", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_api_key, tmp, sizeof(s_api_key) - 1);
        }
        len = sizeof(tmp);
        if (nvs_get_str(nvs, "model_gem", tmp, &len) == ESP_OK && tmp[0]) {
            strncpy(s_model, tmp, sizeof(s_model) - 1);
        }
        nvs_close(nvs);
    }
    
    if (s_api_key[0] == '\0') {
        ESP_LOGW(TAG, "No Gemini API key. Use CLI: set_api_key <KEY>");
    }
    ESP_LOGI(TAG, "Gemini provider initialized (model: %s)", s_model);
    return ESP_OK;
}

static esp_err_t gemini_chat_tools(const char *system_prompt,
                                   cJSON *messages,
                                   const char *tools_json,
                                   llm_response_t *resp)
{
    memset(resp, 0, sizeof(*resp));
    if (s_api_key[0] == '\0') return ESP_ERR_INVALID_STATE;

    cJSON *body = cJSON_CreateObject();
    
    /* System instruction */
    cJSON *sys_inst = cJSON_CreateObject();
    cJSON *sys_parts = cJSON_CreateArray();
    cJSON *sys_p = cJSON_CreateObject();
    cJSON_AddStringToObject(sys_p, "text", system_prompt);
    cJSON_AddItemToArray(sys_parts, sys_p);
    cJSON_AddItemToObject(sys_inst, "parts", sys_parts);
    cJSON_AddItemToObject(body, "system_instruction", sys_inst);

    /* Contents */
    cJSON_AddItemToObject(body, "contents", convert_messages_to_gemini(messages));

    /* Tools */
    if (tools_json) {
        cJSON *tools_arr = cJSON_CreateArray();
        cJSON *tool_decl_wrapper = cJSON_CreateObject();
        cJSON *func_decls = cJSON_CreateArray();
        
        cJSON *in_tools = cJSON_Parse(tools_json);
        if (in_tools && cJSON_IsArray(in_tools)) {
            cJSON *t;
            cJSON_ArrayForEach(t, in_tools) {
                cJSON *decl = cJSON_CreateObject();
                cJSON_AddStringToObject(decl, "name", cJSON_GetObjectItem(t, "name")->valuestring);
                cJSON_AddStringToObject(decl, "description", cJSON_GetObjectItem(t, "description")->valuestring);
                cJSON_AddItemToObject(decl, "parameters", cJSON_Duplicate(cJSON_GetObjectItem(t, "input_schema"), 1));
                cJSON_AddItemToArray(func_decls, decl);
            }
        }
        cJSON_Delete(in_tools);
        
        cJSON_AddItemToObject(tool_decl_wrapper, "function_declarations", func_decls);
        cJSON_AddItemToArray(tools_arr, tool_decl_wrapper);
        cJSON_AddItemToObject(body, "tools", tools_arr);
    }

    char *post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) return ESP_ERR_NO_MEM;

    resp_buf_t rb;
    resp_buf_init(&rb, MIMI_LLM_STREAM_BUF_SIZE);

    int status = 0;
    esp_err_t err = gemini_http_call(post_data, &rb, &status);
    free(post_data);

    if (err != ESP_OK || status != 200) {
        ESP_LOGE(TAG, "API error %d: %.500s", status, rb.data ? rb.data : "");
        resp_buf_free(&rb);
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(rb.data);
    resp_buf_free(&rb);
    if (!root) return ESP_FAIL;

    cJSON *candidates = cJSON_GetObjectItem(root, "candidates");
    if (candidates && cJSON_IsArray(candidates) && cJSON_GetArraySize(candidates) > 0) {
        cJSON *cand = cJSON_GetArrayItem(candidates, 0);
        cJSON *content = cJSON_GetObjectItem(cand, "content");
        cJSON *parts = cJSON_GetObjectItem(content, "parts");

        if (parts && cJSON_IsArray(parts)) {
            size_t total_text = 0;
            cJSON *p;
            cJSON_ArrayForEach(p, parts) {
                cJSON *text = cJSON_GetObjectItem(p, "text");
                if (text && cJSON_IsString(text)) {
                    total_text += strlen(text->valuestring);
                }
            }

            if (total_text > 0) {
                resp->text = calloc(1, total_text + 1);
                cJSON_ArrayForEach(p, parts) {
                    cJSON *text = cJSON_GetObjectItem(p, "text");
                    if (text && cJSON_IsString(text)) {
                        strcat(resp->text, text->valuestring);
                        resp->text_len += strlen(text->valuestring);
                    }
                }
            }

            cJSON_ArrayForEach(p, parts) {
                cJSON *funcCall = cJSON_GetObjectItem(p, "functionCall");
                if (funcCall && resp->call_count < MIMI_MAX_TOOL_CALLS) {
                    llm_tool_call_t *call = &resp->calls[resp->call_count];
                    cJSON *name = cJSON_GetObjectItem(funcCall, "name");
                    cJSON *args = cJSON_GetObjectItem(funcCall, "args");
                    
                    if (name) strncpy(call->name, name->valuestring, sizeof(call->name) - 1);
                    snprintf(call->id, sizeof(call->id), "gem_%d", resp->call_count);
                    
                    if (args) {
                        call->input = cJSON_PrintUnformatted(args);
                        call->input_len = strlen(call->input);
                    }
                    resp->call_count++;
                    resp->tool_use = true;
                }
            }
        }
    }

    cJSON_Delete(root);
    return ESP_OK;
}

const llm_provider_t gemini_provider = {
    .name = "gemini",
    .init = gemini_init,
    .chat_tools = gemini_chat_tools,
};