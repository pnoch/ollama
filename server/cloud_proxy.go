package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/version"
)

const (
	defaultCloudProxyBaseURL      = "https://ollama.com:443"
	defaultCloudProxySigningHost  = "ollama.com"
	cloudProxyBaseURLEnv          = "OLLAMA_CLOUD_BASE_URL"
	legacyCloudAnthropicKey       = "legacy_cloud_anthropic_web_search"
	cloudProxyClientVersionHeader = "X-Ollama-Client-Version"

	// maxDecompressedBodySize limits the size of a decompressed request body
	maxDecompressedBodySize = 20 << 20

	cloudRequestedModelAliasKey = "cloud_requested_model_alias"
	cloudRequestedModelBaseKey  = "cloud_requested_model_base"
)

var (
	cloudProxyBaseURL     = defaultCloudProxyBaseURL
	cloudProxySigningHost = defaultCloudProxySigningHost
	cloudProxySignRequest = signCloudProxyRequest
	cloudProxySigninURL   = signinURL
)

var hopByHopHeaders = map[string]struct{}{
	"connection":          {},
	"content-length":      {},
	"proxy-connection":    {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailer":             {},
	"transfer-encoding":   {},
	"upgrade":             {},
}

func init() {
	baseURL, signingHost, overridden, err := resolveCloudProxyBaseURL(envconfig.Var(cloudProxyBaseURLEnv), mode)
	if err != nil {
		slog.Warn("ignoring cloud base URL override", "env", cloudProxyBaseURLEnv, "error", err)
		return
	}

	cloudProxyBaseURL = baseURL
	cloudProxySigningHost = signingHost

	if overridden {
		slog.Info("cloud base URL override enabled", "env", cloudProxyBaseURLEnv, "url", cloudProxyBaseURL, "mode", mode)
	}
}

func cloudPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost {
			c.Next()
			return
		}

		// Decompress zstd-encoded request bodies so we can inspect the model
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to decompress request body"})
				c.Abort()
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

		// TODO(drifkin): Avoid full-body buffering here for model detection.
		// A future optimization can parse just enough JSON to read "model" (and
		// optionally short-circuit cloud-disabled explicit-cloud requests) while
		// preserving raw passthrough semantics.
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		model, ok := extractModelField(body)
		if !ok {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(model)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}

		c.Set(cloudRequestedModelAliasKey, model)
		c.Set(cloudRequestedModelBaseKey, modelRef.Base)

		if c.Request.URL.Path == "/v1/responses" {
			slog.Warn(
				"cloud proxy responses request",
				"model", model,
				"body_prefix", summarizeResponsesBodyPrefix(body),
			)
		}

		normalizedBody, err := replaceJSONModelField(body, modelRef.Base)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		// TEMP(drifkin): keep Anthropic web search requests on the local middleware
		// path so WebSearchAnthropicWriter can orchestrate follow-up calls.
		if c.Request.URL.Path == "/v1/messages" {
			if hasAnthropicWebSearchTool(body) {
				c.Set(legacyCloudAnthropicKey, true)
				c.Next()
				return
			}
		}

		proxyCloudRequest(c, normalizedBody, disabledOperation)
		c.Abort()
	}
}

func cloudModelPathPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		modelName := strings.TrimSpace(c.Param("model"))
		if modelName == "" {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(modelName)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}

		c.Set(cloudRequestedModelAliasKey, modelName)
		c.Set(cloudRequestedModelBaseKey, modelRef.Base)

		proxyPath := "/v1/models/" + modelRef.Base
		proxyCloudRequestWithPath(c, nil, proxyPath, disabledOperation)
		c.Abort()
	}
}

func proxyCloudJSONRequest(c *gin.Context, payload any, disabledOperation string) {
	// TEMP(drifkin): we currently split out this `WithPath` method because we are
	// mapping `/v1/messages` + web_search to `/api/chat` temporarily. Once we
	// stop doing this, we can inline this method.
	proxyCloudJSONRequestWithPath(c, payload, c.Request.URL.Path, disabledOperation)
}

func proxyCloudJSONRequestWithPath(c *gin.Context, payload any, path string, disabledOperation string) {
	body, err := json.Marshal(payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	proxyCloudRequestWithPath(c, body, path, disabledOperation)
}

func proxyCloudRequest(c *gin.Context, body []byte, disabledOperation string) {
	proxyCloudRequestWithPath(c, body, c.Request.URL.Path, disabledOperation)
}

func proxyCloudRequestWithPath(c *gin.Context, body []byte, path string, disabledOperation string) {
	if disabled, _ := internalcloud.Status(); disabled {
		c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(disabledOperation)})
		return
	}

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     path,
		RawQuery: c.Request.URL.RawQuery,
	})

	outReq, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, targetURL.String(), bytes.NewReader(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	copyProxyRequestHeaders(outReq.Header, c.Request.Header)
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		outReq.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}
	if outReq.Header.Get("Content-Type") == "" && len(body) > 0 {
		outReq.Header.Set("Content-Type", "application/json")
	}

	if err := cloudProxySignRequest(outReq.Context(), outReq); err != nil {
		slog.Warn("cloud proxy signing failed", "error", err)
		writeCloudUnauthorized(c)
		return
	}

	// TODO(drifkin): Add phase-specific proxy timeouts.
	// Connect/TLS/TTFB should have bounded timeouts, but once streaming starts
	// we should not enforce a short total timeout for long-lived responses.
	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	requestedModelAlias, _ := c.Get(cloudRequestedModelAliasKey)
	requestedModelBase, _ := c.Get(cloudRequestedModelBaseKey)
	if err := copyProxyResponseBody(
		c.Writer,
		resp.Body,
		path,
		resp.Header.Get("Content-Type"),
		stringValue(requestedModelAlias),
		stringValue(requestedModelBase),
	); err != nil {
		ctxErr := c.Request.Context().Err()
		if errors.Is(err, context.Canceled) && errors.Is(ctxErr, context.Canceled) {
			slog.Debug(
				"cloud proxy response stream closed by client",
				"path", c.Request.URL.Path,
				"status", resp.StatusCode,
			)
			return
		}

		slog.Warn(
			"cloud proxy response copy failed",
			"path", c.Request.URL.Path,
			"status", resp.StatusCode,
			"request_context_canceled", ctxErr != nil,
			"request_context_err", ctxErr,
			"error", err,
		)
		return
	}
}

func replaceJSONModelField(body []byte, model string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	modelJSON, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = modelJSON

	return json.Marshal(payload)
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}

	reader := r.Body
	if strings.EqualFold(r.Header.Get("Content-Encoding"), "zstd") {
		zstdReader, err := zstd.NewReader(r.Body, zstd.WithDecoderMaxMemory(8<<20))
		if err != nil {
			return nil, err
		}
		defer zstdReader.Close()
		reader = io.NopCloser(zstdReader)
	}

	body, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	if strings.EqualFold(r.Header.Get("Content-Encoding"), "zstd") {
		r.Header.Del("Content-Encoding")
	}
	return body, nil
}

func extractModelField(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}

	raw, ok := payload["model"]
	if !ok {
		return "", false
	}

	var model string
	if err := json.Unmarshal(raw, &model); err != nil {
		return "", false
	}

	model = strings.TrimSpace(model)
	return model, model != ""
}

func hasAnthropicWebSearchTool(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var payload struct {
		Tools []struct {
			Type string `json:"type"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	for _, tool := range payload.Tools {
		if strings.HasPrefix(strings.TrimSpace(tool.Type), "web_search") {
			return true
		}
	}

	return false
}

func writeCloudUnauthorized(c *gin.Context) {
	signinURL, err := cloudProxySigninURL()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": signinURL})
}

func signCloudProxyRequest(ctx context.Context, req *http.Request) error {
	if !strings.EqualFold(req.URL.Hostname(), cloudProxySigningHost) {
		return nil
	}

	ts := strconv.FormatInt(time.Now().Unix(), 10)
	challenge := buildCloudSignatureChallenge(req, ts)
	signature, err := auth.Sign(ctx, []byte(challenge))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", signature)
	return nil
}

func buildCloudSignatureChallenge(req *http.Request, ts string) string {
	query := req.URL.Query()
	query.Set("ts", ts)
	req.URL.RawQuery = query.Encode()

	return fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())
}

func resolveCloudProxyBaseURL(rawOverride string, runMode string) (baseURL string, signingHost string, overridden bool, err error) {
	baseURL = defaultCloudProxyBaseURL
	signingHost = defaultCloudProxySigningHost

	rawOverride = strings.TrimSpace(rawOverride)
	if rawOverride == "" {
		return baseURL, signingHost, false, nil
	}

	u, err := url.Parse(rawOverride)
	if err != nil {
		return "", "", false, fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", "", false, fmt.Errorf("invalid URL: scheme and host are required")
	}
	if u.User != nil {
		return "", "", false, fmt.Errorf("invalid URL: userinfo is not allowed")
	}
	if u.Path != "" && u.Path != "/" {
		return "", "", false, fmt.Errorf("invalid URL: path is not allowed")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return "", "", false, fmt.Errorf("invalid URL: query and fragment are not allowed")
	}

	host := u.Hostname()
	if host == "" {
		return "", "", false, fmt.Errorf("invalid URL: host is required")
	}

	loopback := isLoopbackHost(host)
	if runMode == gin.ReleaseMode && !loopback {
		return "", "", false, fmt.Errorf("non-loopback cloud override is not allowed in release mode")
	}
	if !loopback && !strings.EqualFold(u.Scheme, "https") {
		return "", "", false, fmt.Errorf("non-loopback cloud override must use https")
	}

	u.Path = ""
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""

	return u.String(), strings.ToLower(host), true, nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func copyProxyRequestHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseBody(dst http.ResponseWriter, src io.Reader, path, contentType, modelAlias, modelBase string) error {
	if path == "/v1/responses" {
		reader := bufio.NewReader(src)
		format, sample := detectResponsesProxyFormat(reader, contentType)
		slog.Warn(
			"cloud proxy responses format detected",
			"path", path,
			"content_type", contentType,
			"detected_format", format,
			"body_prefix", sample,
		)
		switch format {
		case "event-stream":
			return copySanitizedResponsesEventStream(dst, reader, modelAlias, modelBase)
		case "json":
			return copySanitizedResponsesJSON(dst, reader, modelAlias, modelBase)
		}
	}

	return copyProxyStream(dst, src)
}

func copyProxyStream(dst http.ResponseWriter, src io.Reader) error {
	flusher, canFlush := dst.(http.Flusher)
	buf := make([]byte, 32*1024)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			if canFlush {
				// TODO(drifkin): Consider conditional flushing so non-streaming
				// responses don't flush every write and can optimize throughput.
				flusher.Flush()
			}
		}

		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

func detectResponsesProxyFormat(r *bufio.Reader, contentType string) (string, string) {
	contentType = strings.ToLower(contentType)
	if strings.Contains(contentType, "text/event-stream") {
		return "event-stream", ""
	}

	peek, err := r.Peek(4096)
	sample := summarizeResponsesBodyPrefix(peek)
	if err == nil || errors.Is(err, bufio.ErrBufferFull) || errors.Is(err, io.EOF) {
		if looksLikeResponsesEventStream(peek) {
			return "event-stream", sample
		}

		trimmed := bytes.TrimSpace(peek)
		if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') {
			return "json", sample
		}
	}

	if strings.Contains(contentType, "application/json") {
		return "json", sample
	}

	return "", sample
}

func copySanitizedResponsesJSON(dst http.ResponseWriter, src io.Reader, modelAlias, modelBase string) error {
	body, err := io.ReadAll(src)
	if err != nil {
		return err
	}

	sanitized, err := sanitizeResponsesPayload(body, modelAlias, modelBase)
	if err != nil {
		if looksLikeResponsesEventStream(body) {
			return copySanitizedResponsesEventStream(dst, bytes.NewReader(body), modelAlias, modelBase)
		}
		slog.Warn(
			"cloud proxy responses json sanitize failed",
			"body_prefix", summarizeResponsesBodyPrefix(body),
			"error", err,
		)
		return err
	}

	_, err = dst.Write(sanitized)
	return err
}

func copySanitizedResponsesEventStream(dst http.ResponseWriter, src io.Reader, modelAlias, modelBase string) error {
	flusher, canFlush := dst.(http.Flusher)
	reader := bufio.NewReader(src)
	var currentEventType string
	var skipCurrentEvent bool
	var pendingEventLine []byte
	var sawDoneSentinel bool
	var sawAssistantOutputItemDone bool

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if bytes.HasPrefix(line, []byte("event: ")) {
				currentEventType = strings.TrimSpace(string(bytes.TrimPrefix(line, []byte("event: "))))
				skipCurrentEvent = shouldSkipResponsesEventType(currentEventType)
				if currentEventType == "done" {
					sawDoneSentinel = true
				}
				pendingEventLine = append(pendingEventLine[:0], line...)
				if skipCurrentEvent {
					goto nextLine
				}
				goto nextLine
			}

			out := line
			if bytes.HasPrefix(line, []byte("data: ")) {
				if sanitized, ok, skip, sanitizeErr := sanitizeResponsesEventData(line, currentEventType, modelAlias, modelBase); sanitizeErr != nil {
					slog.Warn(
						"cloud proxy responses event sanitize failed",
						"line_prefix", summarizeResponsesBodyPrefix(line),
						"error", sanitizeErr,
					)
					return sanitizeErr
				} else if skip {
					skipCurrentEvent = true
					goto nextLine
				} else if ok {
					if currentEventType == "response.output_item.done" && responsesEventContainsAssistantMessage(sanitized) {
						sawAssistantOutputItemDone = true
					}
					if currentEventType == "response.completed" {
						syntheticDoneEvents, synthErr := buildSyntheticAssistantOutputItemDoneEvents(sanitized)
						if synthErr != nil {
							return synthErr
						}
						if !sawAssistantOutputItemDone && len(syntheticDoneEvents) > 0 {
							if len(pendingEventLine) > 0 && !bytes.Equal(line, pendingEventLine) {
								if _, writeErr := dst.Write(pendingEventLine); writeErr != nil {
									return writeErr
								}
								pendingEventLine = pendingEventLine[:0]
							}
							for _, synthetic := range syntheticDoneEvents {
								if _, writeErr := dst.Write([]byte("event: response.output_item.done\n")); writeErr != nil {
									return writeErr
								}
								if _, writeErr := dst.Write(synthetic); writeErr != nil {
									return writeErr
								}
								if _, writeErr := dst.Write([]byte("\n")); writeErr != nil {
									return writeErr
								}
								if canFlush {
									flusher.Flush()
								}
							}
							sawAssistantOutputItemDone = true
						}
					}
					out = sanitized
				}
			}

			if len(bytes.TrimSpace(line)) == 0 {
				if skipCurrentEvent {
					skipCurrentEvent = false
					currentEventType = ""
					pendingEventLine = pendingEventLine[:0]
					goto nextLine
				}
				currentEventType = ""
			}

			if skipCurrentEvent {
				goto nextLine
			}

			if len(pendingEventLine) > 0 && !bytes.Equal(line, pendingEventLine) {
				if _, writeErr := dst.Write(pendingEventLine); writeErr != nil {
					return writeErr
				}
				pendingEventLine = pendingEventLine[:0]
			}

			if _, writeErr := dst.Write(out); writeErr != nil {
				return writeErr
			}
			if canFlush {
				flusher.Flush()
			}
		}

	nextLine:

		if err != nil {
			if err == io.EOF {
				if !sawDoneSentinel {
					if _, writeErr := dst.Write([]byte("event: done\n")); writeErr != nil {
						return writeErr
					}
					if _, writeErr := dst.Write([]byte("data: [DONE]\n\n")); writeErr != nil {
						return writeErr
					}
					if canFlush {
						flusher.Flush()
					}
				}
				if len(pendingEventLine) > 0 && !skipCurrentEvent {
					if _, writeErr := dst.Write(pendingEventLine); writeErr != nil {
						return writeErr
					}
					if canFlush {
						flusher.Flush()
					}
				}
				return nil
			}
			return err
		}
	}
}

func sanitizeResponsesEventData(line []byte, eventType, modelAlias, modelBase string) ([]byte, bool, bool, error) {
	payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data: ")))
	if len(payload) == 0 {
		return nil, false, false, nil
	}

	first := payload[0]
	if first != '{' && first != '[' {
		return nil, false, false, nil
	}

	var parsed any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return nil, false, false, err
	}
	if shouldSkipResponsesEventPayload(eventType, parsed) {
		return nil, false, true, nil
	}

	sanitizedValue, drop := sanitizeResponsesValue(parsed, modelAlias, modelBase)
	if drop {
		return nil, false, true, nil
	}
	sanitized, err := json.Marshal(sanitizedValue)
	if err != nil {
		return nil, false, false, err
	}

	out := append([]byte("data: "), sanitized...)
	out = append(out, '\n')
	return out, true, false, nil
}

func responsesEventContainsAssistantMessage(line []byte) bool {
	payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data: ")))
	var parsed map[string]any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return false
	}
	item, _ := parsed["item"].(map[string]any)
	if item == nil {
		return false
	}
	itemType, _ := item["type"].(string)
	role, _ := item["role"].(string)
	return itemType == "message" && role == "assistant"
}

func buildSyntheticAssistantOutputItemDoneEvents(line []byte) ([][]byte, error) {
	payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data: ")))
	var parsed map[string]any
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return nil, err
	}

	response, _ := parsed["response"].(map[string]any)
	if response == nil {
		return nil, nil
	}
	output, _ := response["output"].([]any)
	if len(output) == 0 {
		return nil, nil
	}

	synthetic := make([][]byte, 0, len(output))
	for index, rawItem := range output {
		item, _ := rawItem.(map[string]any)
		if item == nil {
			continue
		}
		itemType, _ := item["type"].(string)
		role, _ := item["role"].(string)
		if itemType != "message" || role != "assistant" {
			continue
		}
		event := map[string]any{
			"type":         "response.output_item.done",
			"output_index": index,
			"item":         item,
		}
		marshaled, err := json.Marshal(event)
		if err != nil {
			return nil, err
		}
		line := append([]byte("data: "), marshaled...)
		line = append(line, '\n')
		synthetic = append(synthetic, line)
	}

	return synthetic, nil
}

func looksLikeResponsesEventStream(body []byte) bool {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return false
	}

	return bytes.HasPrefix(trimmed, []byte("event:")) ||
		bytes.HasPrefix(trimmed, []byte("data:")) ||
		bytes.Contains(trimmed, []byte("\nevent:")) ||
		bytes.Contains(trimmed, []byte("\ndata:"))
}

func summarizeResponsesBodyPrefix(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	if len(body) > 160 {
		body = body[:160]
	}

	replacer := strings.NewReplacer("\r", "\\r", "\n", "\\n", "\t", "\\t")
	return replacer.Replace(string(body))
}

func sanitizeResponsesPayload(body []byte, modelAlias, modelBase string) ([]byte, error) {
	var payload any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	sanitized, drop := sanitizeResponsesValue(payload, modelAlias, modelBase)
	if drop {
		return json.Marshal(map[string]any{})
	}
	return json.Marshal(sanitized)
}

func sanitizeResponsesValue(v any, modelAlias, modelBase string) (any, bool) {
	switch x := v.(type) {
	case map[string]any:
		if itemType, _ := x["type"].(string); itemType == "reasoning" {
			return nil, true
		}
		if objectType, _ := x["object"].(string); objectType == "response" {
			ensureCloudResponseShape(x)
		}
		if itemType, _ := x["type"].(string); itemType == "message" {
			if role, _ := x["role"].(string); role == "assistant" {
				if _, ok := x["phase"]; !ok {
					x["phase"] = "final_answer"
				}
			}
		}
		if modelValue, _ := x["model"].(string); modelAlias != "" && modelBase != "" && modelValue == modelBase {
			x["model"] = modelAlias
		}
		for key, value := range x {
			sanitized, drop := sanitizeResponsesValue(value, modelAlias, modelBase)
			if drop {
				delete(x, key)
				continue
			}
			x[key] = sanitized
		}
		if responseValue, ok := x["response"].(map[string]any); ok {
			ensureCloudResponseShape(responseValue)
			x["response"] = responseValue
		}
		if outputIndex, ok := x["output_index"]; ok {
			switch n := outputIndex.(type) {
			case float64:
				if n > 0 {
					x["output_index"] = n - 1
				}
			case int:
				if n > 0 {
					x["output_index"] = n - 1
				}
			}
		}
		return x, false
	case []any:
		filtered := make([]any, 0, len(x))
		for _, value := range x {
			sanitized, drop := sanitizeResponsesValue(value, modelAlias, modelBase)
			if drop {
				continue
			}
			filtered = append(filtered, sanitized)
		}
		return filtered, false
	}
	return v, false
}

func ensureCloudResponseShape(x map[string]any) {
	if reasoning, ok := x["reasoning"]; !ok || reasoning == nil {
		x["reasoning"] = map[string]any{
			"effort":  "none",
			"summary": nil,
		}
	}
	text, _ := x["text"].(map[string]any)
	if text == nil {
		text = map[string]any{}
		x["text"] = text
	}
	if _, ok := text["format"]; !ok {
		text["format"] = map[string]any{"type": "text"}
	}
	if _, ok := text["verbosity"]; !ok {
		text["verbosity"] = "medium"
	}
	if _, ok := x["prompt_cache_retention"]; !ok {
		x["prompt_cache_retention"] = nil
	}
	if _, ok := x["user"]; !ok {
		x["user"] = nil
	}
	x["store"] = true
}

func shouldSkipResponsesEventType(eventType string) bool {
	return strings.HasPrefix(eventType, "response.reasoning_")
}

func shouldSkipResponsesEventPayload(eventType string, payload any) bool {
	if !strings.HasPrefix(eventType, "response.output_item.") {
		return false
	}
	m, ok := payload.(map[string]any)
	if !ok {
		return false
	}
	item, ok := m["item"].(map[string]any)
	if !ok {
		return false
	}
	itemType, _ := item["type"].(string)
	return itemType == "reasoning"
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func isHopByHopHeader(name string) bool {
	_, ok := hopByHopHeaders[strings.ToLower(name)]
	return ok
}

func connectionHeaderTokens(header http.Header) map[string]struct{} {
	tokens := map[string]struct{}{}
	for _, raw := range header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			token = strings.TrimSpace(strings.ToLower(token))
			if token == "" {
				continue
			}
			tokens[token] = struct{}{}
		}
	}
	return tokens
}

func isConnectionTokenHeader(name string, tokens map[string]struct{}) bool {
	if len(tokens) == 0 {
		return false
	}
	_, ok := tokens[strings.ToLower(name)]
	return ok
}
