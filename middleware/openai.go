package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
)

// maxDecompressedBodySize limits the size of a decompressed request body
const maxDecompressedBodySize = 20 << 20

type BaseWriter struct {
	gin.ResponseWriter
}

type ChatWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	toolCallSent  bool
	BaseWriter
}

type CompleteWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	BaseWriter
}

type ListWriter struct {
	BaseWriter
}

type RetrieveWriter struct {
	BaseWriter
	model string
}

type EmbedWriter struct {
	BaseWriter
	model          string
	encodingFormat string
}

func (w *BaseWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	if err := json.Unmarshal(data, &serr); err != nil {
		// If the error response isn't valid JSON, use the raw bytes as the
		// error message rather than surfacing a confusing JSON parse error.
		serr.ErrorMessage = string(data)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(w.ResponseWriter.Status(), serr.Error())); err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// chat chunk
	if w.stream {
		chunks := openai.ToChunks(w.id, chatResponse, w.toolCallSent)
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			d, err := json.Marshal(c)
			if err != nil {
				return 0, err
			}
			if !w.toolCallSent && len(c.Choices) > 0 && len(c.Choices[0].Delta.ToolCalls) > 0 {
				w.toolCallSent = true
			}
			_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
			if err != nil {
				return 0, err
			}
		}

		if chatResponse.Done {
			c := openai.ToChunk(w.id, chatResponse, w.toolCallSent)
			if len(chunks) > 0 {
				c = chunks[len(chunks)-1]
			} else {
				slog.Warn("ToChunks returned no chunks; falling back to ToChunk for usage chunk", "id", w.id, "model", chatResponse.Model)
			}
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsage(chatResponse)
				c.Usage = &u
				c.Choices = []openai.ChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *CompleteWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	err := json.Unmarshal(data, &generateResponse)
	if err != nil {
		return 0, err
	}

	// completion chunk
	if w.stream {
		c := openai.ToCompleteChunk(w.id, generateResponse)
		if w.streamOptions != nil && w.streamOptions.IncludeUsage {
			c.Usage = &openai.Usage{}
		}
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if generateResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsageGenerate(generateResponse)
				c.Usage = &u
				c.Choices = []openai.CompleteChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToCompletion(w.id, generateResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *CompleteWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *ListWriter) writeResponse(data []byte) (int, error) {
	var listResponse api.ListResponse
	err := json.Unmarshal(data, &listResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToListCompletion(listResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ListWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *RetrieveWriter) writeResponse(data []byte) (int, error) {
	var showResponse api.ShowResponse
	err := json.Unmarshal(data, &showResponse)
	if err != nil {
		return 0, err
	}

	// retrieve completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToModel(showResponse, w.model))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *RetrieveWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *EmbedWriter) writeResponse(data []byte) (int, error) {
	var embedResponse api.EmbedResponse
	err := json.Unmarshal(data, &embedResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToEmbeddingList(w.model, embedResponse, w.encodingFormat))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *EmbedWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		w := &ListWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w

		c.Next()
	}
}

func RetrieveMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.ShowRequest{Name: c.Param("model")}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &RetrieveWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			model:      c.Param("model"),
		}

		c.Writer = w

		c.Next()
	}
}

func CompletionsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.CompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		genReq, err := openai.FromCompleteRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &CompleteWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("cmpl-%s", uuid.New().String()[:8]),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w
		c.Next()
	}
}

func EmbeddingsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.EmbedRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Validate encoding_format parameter
		if req.EncodingFormat != "" {
			if !strings.EqualFold(req.EncodingFormat, "float") && !strings.EqualFold(req.EncodingFormat, "base64") {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("Invalid value for 'encoding_format' = %s. Supported values: ['float', 'base64'].", req.EncodingFormat)))
				return
			}
		}

		if req.Input == "" {
			req.Input = []string{""}
		}

		if req.Input == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		if v, ok := req.Input.([]any); ok && len(v) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.EmbedRequest{Model: req.Model, Input: req.Input, Dimensions: req.Dimensions}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &EmbedWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			model:          req.Model,
			encodingFormat: req.EncodingFormat,
		}

		c.Writer = w

		c.Next()
	}
}

func ChatMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ChatCompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer

		chatReq, err := openai.FromChatRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ChatWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("chatcmpl-%s", uuid.New().String()[:8]),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w

		c.Next()
	}
}

type ResponsesWriter struct {
	BaseWriter
	converter  *openai.ResponsesStreamConverter
	model      string
	stream     bool
	responseID string
	itemID     string
	request    openai.ResponsesRequest
}

func (w *ResponsesWriter) writeEvent(eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, d)))
	if err != nil {
		return err
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (w *ResponsesWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	if w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		events := w.converter.Process(chatResponse)
		for _, event := range events {
			if err := w.writeEvent(event.Event, event.Data); err != nil {
				return 0, err
			}
		}
		return len(data), nil
	}

	// Non-streaming response
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

// writeResponseWithWebSearch builds a non-streaming Responses API response that
// includes a web_search_call output item before the final assistant message.
// It also extracts url_citation annotations from the response text and injects
// them into the output_text content per the OpenAI Responses API spec.
func (w *ResponsesWriter) writeResponseWithWebSearch(data []byte, wsItemID, query string, results []map[string]any) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	// Inject url_citation annotations into output_text content items.
	injectWebSearchCitations(response.Output, results)
	// Prepend the web_search_call output item.
	wsItem := openai.ResponsesOutputItem{
		ID:     wsItemID,
		Type:   "web_search_call",
		Status: "completed",
	}
	response.Output = append([]openai.ResponsesOutputItem{wsItem}, response.Output...)
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

// injectWebSearchCitations scans each output_text content item in the output
// list for citation markers (e.g. [1], [Source 2]) and replaces the empty
// annotations slice with url_citation objects that point to the search results.
func injectWebSearchCitations(output []openai.ResponsesOutputItem, results []map[string]any) {
	for i := range output {
		if output[i].Type != "message" {
			continue
		}
		for j := range output[i].Content {
			if output[i].Content[j].Type != "output_text" {
				continue
			}
			citations := openai.ExtractWebSearchCitations(output[i].Content[j].Text, results)
			if len(citations) > 0 {
				anns := make([]any, len(citations))
				for k, c := range citations {
					anns[k] = c
				}
				output[i].Content[j].Annotations = anns
			}
		}
	}
}

// writeResponseWithFileSearch builds a non-streaming Responses API response
// with a file_search_call output item prepended to the output list.
func (w *ResponsesWriter) writeResponseWithFileSearch(data []byte, fsItemID, query string, chunks []FileChunkResult) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	// Prepend the file_search_call output item.
	fsItem := openai.ResponsesOutputItem{
		ID:     fsItemID,
		Type:   "file_search_call",
		Status: "completed",
	}
	response.Output = append([]openai.ResponsesOutputItem{fsItem}, response.Output...)
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

func (w *ResponsesWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}
	return w.writeResponse(data)
}

func ResponsesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to decompress zstd body"))
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		var modelOnly struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &modelOnly); err == nil && strings.HasSuffix(strings.TrimSpace(modelOnly.Model), ":cloud") {
			c.Next()
			return
		}

		var req openai.ResponsesRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// background mode is not supported for local models; return a clear error
		// rather than silently ignoring the flag and running synchronously.
		if req.Background {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest,
				"background mode is not supported for local models; use a cloud model or omit the background field"))
			return
		}

		chatReq, err := openai.FromResponsesRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Check if client requested streaming (defaults to false)
		streamRequested := req.Stream != nil && *req.Stream

		// Pass streaming preference to the underlying chat request
		chatReq.Stream = &streamRequested

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		responseID := fmt.Sprintf("resp_%s", uuid.New().String())
		itemID := fmt.Sprintf("msg_%s", uuid.New().String()[:8])

		w := &ResponsesWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			converter:  openai.NewResponsesStreamConverter(responseID, itemID, req.Model, req),
			model:      req.Model,
			stream:     streamRequested,
			responseID: responseID,
			itemID:     itemID,
			request:    req,
		}

		// Set headers based on streaming mode
		if streamRequested {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		// If the request includes a web_search_preview built-in tool, wrap the
		// writer with the agent loop that executes searches server-side and
		// emits web_search_call events. This makes web_search_preview work
		// identically for local models as it does for cloud models.
			if wsTool, hasWS := openai.HasWebSearchPreview(req.Tools); hasWS {
				c.Writer = &ResponsesWebSearchWriter{
					BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
					inner:          w,
					req:            req,
					chatReq:        chatReq,
					webSearchTool:  wsTool,
				}
		} else if fsTool, hasFS := openai.HasFileSearch(req.Tools); hasFS {
			// If the request includes a file_search built-in tool, wrap the writer
			// with the agent loop that queries the local vector store and emits
			// file_search_call events. The FileSearcher and FileEmbedder are
			// injected via gin context keys set by the server before this
			// middleware runs (see Server.RegisterFileSearchDeps).
			var searcher FileSearcher
			var embedder FileEmbedder
			if s, ok := c.Get("file_searcher"); ok {
				// Try direct FileSearcher first (e.g. in tests), then fall back to
				// the []any variant used by the server adapter to avoid import cycles.
				if fs, ok := s.(FileSearcher); ok {
					searcher = fs
				} else if fa, ok := s.(fileSearcherAny); ok {
					searcher = &fileSearcherAnyAdapter{inner: fa}
				}
			}
			if e, ok := c.Get("file_embedder"); ok {
				embedder, _ = e.(FileEmbedder)
			}
			if searcher != nil && embedder != nil {
				c.Writer = &ResponsesFileSearchWriter{
					BaseWriter: BaseWriter{ResponseWriter: c.Writer},
					inner:      w,
					req:        req,
					chatReq:    chatReq,
					vsIDs:      fsTool.VectorStoreIDs,
					maxResults: fsTool.MaxNumResults,
					embedder:   embedder,
					searcher:   searcher,
				}
			} else {
				// No vector store injected (e.g. test environment) — fall through.
				c.Writer = w
			}
		} else {
			c.Writer = w
		}
		c.Next()
	}
}

type ImageWriter struct {
	BaseWriter
}

func (w *ImageWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	if err := json.Unmarshal(data, &generateResponse); err != nil {
		return 0, err
	}

	// Only write response when done with image
	if generateResponse.Done && generateResponse.Image != "" {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		return len(data), json.NewEncoder(w.ResponseWriter).Encode(openai.ToImageGenerationResponse(generateResponse))
	}

	return len(data), nil
}

func (w *ImageWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func ImageGenerationsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ImageGenerationRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(openai.FromImageGenerationRequest(req)); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w
		c.Next()
	}
}

func ImageEditsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ImageEditRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		if req.Image == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "image is required"))
			return
		}

		genReq, err := openai.FromImageEditRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w
		c.Next()
	}
}

// ResponsesWebSearchWriter intercepts Responses API responses that contain a
// web_search tool call, executes the search server-side (via ollama.com/api/web_search),
// feeds the results back to the model as a tool message, and emits the correct
// Responses API streaming events (web_search_call, then the final text response).
//
// This makes web_search_preview work identically for both cloud and local models:
// the cloud proxy forwards the request verbatim; local models go through this writer.
type ResponsesWebSearchWriter struct {
	BaseWriter
	inner         *ResponsesWriter
	req           openai.ResponsesRequest
	chatReq       *api.ChatRequest
	webSearchTool openai.ResponsesTool
}

const maxResponsesWebSearchLoops = 3

func (w *ResponsesWebSearchWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.inner.writeError(data)
	}
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	wsCall, hasWebSearch, _ := findResponsesWebSearchCall(chatResponse.Message.ToolCalls)
	if !hasWebSearch {
		// No web_search call — pass through to the normal ResponsesWriter.
		return w.inner.writeResponse(data)
	}

	// Run the agent loop synchronously (non-streaming and streaming both handled here).
	ctx := w.inner.request.Stream != nil // just used as a bool below
	_ = ctx
	return w.runLoop(chatResponse, wsCall)
}

// webSearchCitationInstruction is appended to the last system/developer message
// (or prepended as a new system message) in the web search follow-up call.
// It instructs the model to include source URLs verbatim so that
// ExtractWebSearchCitations can match them and produce url_citation annotations.
const webSearchCitationInstruction = `When you reference information from the search results, include the source URL verbatim somewhere in your response text (e.g. "According to https://example.com/page, ..."). This allows citations to be linked automatically.`

// injectCitationInstruction appends the citation instruction to the last
// system or developer message in msgs, or prepends a new system message if
// none exists. The instruction text is taken from OLLAMA_WEB_SEARCH_CITATION_PROMPT
// if set, otherwise the built-in default is used.
func injectCitationInstruction(msgs []api.Message) []api.Message {
	instruction := envconfig.WebSearchCitationPrompt()
	if instruction == "" {
		instruction = webSearchCitationInstruction
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "system" || msgs[i].Role == "developer" {
			msgs[i].Content = strings.TrimRight(msgs[i].Content, " \n") + "\n\n" + instruction
			return msgs
		}
	}
	return append([]api.Message{{Role: "system", Content: instruction}}, msgs...)
}

func (w *ResponsesWebSearchWriter) runLoop(initialResp api.ChatResponse, initialCall api.ToolCall) (int, error) {
	followUpMessages := make([]api.Message, 0, len(w.chatReq.Messages)+maxResponsesWebSearchLoops*2)
	followUpMessages = append(followUpMessages, w.chatReq.Messages...)
	// Inject a citation instruction so the model includes source URLs verbatim.
	// This dramatically increases the hit rate of ExtractWebSearchCitations.
	followUpMessages = injectCitationInstruction(followUpMessages)
	followUpTools := append(api.Tools(nil), w.chatReq.Tools...)

	currentResp := initialResp
	currentCall := initialCall

	stream := w.req.Stream != nil && *w.req.Stream

	// For streaming: emit response.created + response.in_progress first.
	// The converter handles this on the first Process() call, but since we are
	// taking over the write path we need to prime it.
	if stream {
		w.inner.converter.ResetFirstWrite()
	}

	for loop := 1; loop <= maxResponsesWebSearchLoops; loop++ {
		query := responsesExtractQuery(&currentCall)
		if query == "" {
			break
		}

		// Execute the web search.
		results, err := responsesWebSearch(query, w.webSearchTool.MaxResults)
		if err != nil {
			// On search failure, fall back to passing the original response through.
			break
		}

		wsItemID := fmt.Sprintf("ws_%d_%d", loop, len(query))

		// Emit web_search_call events (streaming only; non-streaming builds them at end).
		if stream {
			events := w.inner.converter.WebSearchCallEvents(wsItemID, query, results)
			for _, ev := range events {
				if err := w.inner.writeEvent(ev.Event, ev.Data); err != nil {
					return 0, err
				}
			}
		}

		// Build follow-up messages: assistant tool call + tool result.
		assistantMsg := api.Message{
			Role:      "assistant",
			ToolCalls: []api.ToolCall{currentCall},
		}
		if currentResp.Message.Content != "" {
			assistantMsg.Content = currentResp.Message.Content
		}
		toolResultMsg := api.Message{
			Role:       "tool",
			Content:    responsesFormatSearchResults(results),
			ToolCallID: currentCall.ID,
		}
		followUpMessages = append(followUpMessages, assistantMsg, toolResultMsg)

		// Call /api/chat for the follow-up.
		followUpResp, err := responsesCallFollowUp(w.chatReq.Model, followUpMessages, followUpTools, w.chatReq.Options)
		if err != nil {
			break
		}

		// Check if the follow-up also wants to search.
		nextCall, hasMore, _ := findResponsesWebSearchCall(followUpResp.Message.ToolCalls)
		if !hasMore {
			// Final response — emit it.
			if stream {
				// Emit the final text response events.
				finalData, err := json.Marshal(followUpResp)
				if err != nil {
					return 0, err
				}
				return w.inner.writeResponse(finalData)
			}
			// Non-streaming: build response with web_search_call items prepended.
			finalData, err := json.Marshal(followUpResp)
			if err != nil {
				return 0, err
			}
			return w.inner.writeResponseWithWebSearch(finalData, wsItemID, query, results)
		}
		currentResp = followUpResp
		currentCall = nextCall
	}

	// Fallback: emit the last response as-is.
	fallbackData, err := json.Marshal(currentResp)
	if err != nil {
		return 0, err
	}
	return w.inner.writeResponse(fallbackData)
}

// findResponsesWebSearchCall finds the first web_search tool call in a list.
func findResponsesWebSearchCall(toolCalls []api.ToolCall) (api.ToolCall, bool, bool) {
	var wsCall api.ToolCall
	hasWebSearch := false
	hasOther := false
	for _, tc := range toolCalls {
		if tc.Function.Name == "web_search" {
			if !hasWebSearch {
				wsCall = tc
				hasWebSearch = true
			}
			continue
		}
		hasOther = true
	}
	return wsCall, hasWebSearch, hasOther
}

// responsesExtractQuery extracts the "query" argument from a web_search tool call.
func responsesExtractQuery(tc *api.ToolCall) string {
	q, ok := tc.Function.Arguments.Get("query")
	if !ok {
		return ""
	}
	if s, ok := q.(string); ok {
		return s
	}
	return ""
}

// responsesWebSearch calls ollama.com/api/web_search and returns results as
// []map[string]any for embedding in Responses API events.
func responsesWebSearch(query string, maxResults int) ([]map[string]any, error) {
	type searchReq struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results,omitempty"`
	}
	type searchResult struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Content string `json:"content"`
	}
	type searchResp struct {
		Results []searchResult `json:"results"`
	}

	if maxResults <= 0 {
		maxResults = int(envconfig.WebSearchMaxResults())
	}
	body, err := json.Marshal(searchReq{Query: query, MaxResults: maxResults})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, "https://ollama.com/api/web_search", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("web search returned status %d", resp.StatusCode)
	}

	var sr searchResp
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, err
	}

	results := make([]map[string]any, 0, len(sr.Results))
	for _, r := range sr.Results {
		results = append(results, map[string]any{
			"title":   r.Title,
			"url":     r.URL,
			"content": r.Content,
		})
	}
	return results, nil
}

// responsesFormatSearchResults formats search results as a plain-text tool message.
func responsesFormatSearchResults(results []map[string]any) string {
	var sb strings.Builder
	for i, r := range results {
		title, _ := r["title"].(string)
		url, _ := r["url"].(string)
		content, _ := r["content"].(string)
		fmt.Fprintf(&sb, "%d. %s\n   URL: %s\n", i+1, title, url)
		if content != "" {
			runes := []rune(content)
			if len(runes) > 300 {
				content = string(runes[:300]) + "..."
			}
			fmt.Fprintf(&sb, "   %s\n", content)
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// responsesCallFollowUp makes a non-streaming /api/chat call for the web search follow-up.
func responsesCallFollowUp(model string, messages []api.Message, tools api.Tools, options map[string]any) (api.ChatResponse, error) {
	streaming := false
	req := api.ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   &streaming,
		Tools:    tools,
		Options:  options,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return api.ChatResponse{}, err
	}
	chatURL := "http://localhost:11434/api/chat"
	httpReq, err := http.NewRequest(http.MethodPost, chatURL, bytes.NewReader(body))
	if err != nil {
		return api.ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return api.ChatResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return api.ChatResponse{}, fmt.Errorf("follow-up /api/chat returned status %d", resp.StatusCode)
	}
	var chatResp api.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return api.ChatResponse{}, err
	}
	return chatResp, nil
}

// ─── File Search Agent Loop ───────────────────────────────────────────────────

// FileChunkResult is a single search result from a vector store. It mirrors
// vectorstore.ChunkResult but is defined here to avoid a circular import
// (server → middleware → server/vectorstore → server).
type FileChunkResult struct {
	FileID     string  `json:"file_id"`
	Filename   string  `json:"filename"`
	Text       string  `json:"text"`
	Score      float64 `json:"score"`
	ChunkIndex int     `json:"chunk_index"`
}

// FileSearcher is the interface the middleware uses to query a vector store.
// It is satisfied by a thin adapter on *vectorstore.Store defined in
// server/routes_vectorstore.go.
type FileSearcher interface {
	SearchChunks(vsIDs []string, queryVec []float32, maxResults int) ([]FileChunkResult, error)
}

// fileSearcherAny is an internal adapter that accepts a FileSearcher-like
// interface where SearchChunks returns []any instead of []FileChunkResult.
// This lets the server adapter avoid importing the middleware package.
type fileSearcherAny interface {
	SearchChunks(vsIDs []string, queryVec []float32, maxResults int) ([]any, error)
}

// fileSearcherAnyAdapter wraps a fileSearcherAny and converts its results
// to []FileChunkResult for use by ResponsesFileSearchWriter.
type fileSearcherAnyAdapter struct{ inner fileSearcherAny }

func (a *fileSearcherAnyAdapter) SearchChunks(vsIDs []string, queryVec []float32, maxResults int) ([]FileChunkResult, error) {
	raw, err := a.inner.SearchChunks(vsIDs, queryVec, maxResults)
	if err != nil {
		return nil, err
	}
	out := make([]FileChunkResult, 0, len(raw))
	for _, r := range raw {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		var cr FileChunkResult
		cr.FileID, _ = m["file_id"].(string)
		cr.Filename, _ = m["filename"].(string)
		cr.Text, _ = m["text"].(string)
		cr.Score, _ = m["score"].(float64)
		if ci, ok := m["chunk_index"].(int64); ok {
			cr.ChunkIndex = int(ci)
		} else if ci, ok := m["chunk_index"].(float64); ok {
			cr.ChunkIndex = int(ci)
		}
		out = append(out, cr)
	}
	return out, nil
}

// FileEmbedder is the interface the middleware uses to embed a query string.
type FileEmbedder interface {
	EmbedText(text string) ([]float32, error)
}

// ResponsesFileSearchWriter intercepts Responses API responses that contain a
// file_search tool call, queries the local vector store, feeds the results back
// to the model as a tool message, and emits the correct Responses API streaming
// events (file_search_call, then the final text response).
//
// This makes file_search work identically for both cloud and local models.
type ResponsesFileSearchWriter struct {
	BaseWriter
	inner      *ResponsesWriter
	req        openai.ResponsesRequest
	chatReq    *api.ChatRequest
	vsIDs      []string
	maxResults int
	embedder   FileEmbedder
	searcher   FileSearcher
}

const maxResponsesFileSearchLoops = 3

func (w *ResponsesFileSearchWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.inner.writeError(data)
	}
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}
	fsCall, hasFS := findResponsesFileSearchCall(chatResponse.Message.ToolCalls)
	if !hasFS {
		return w.inner.writeResponse(data)
	}
	return w.runFileSearchLoop(chatResponse, fsCall)
}

func (w *ResponsesFileSearchWriter) runFileSearchLoop(initialResp api.ChatResponse, initialCall api.ToolCall) (int, error) {
	followUpMessages := make([]api.Message, 0, len(w.chatReq.Messages)+maxResponsesFileSearchLoops*2)
	followUpMessages = append(followUpMessages, w.chatReq.Messages...)
	followUpTools := append(api.Tools(nil), w.chatReq.Tools...)

	currentResp := initialResp
	currentCall := initialCall
	stream := w.req.Stream != nil && *w.req.Stream

	if stream {
		w.inner.converter.ResetFirstWrite()
	}

	for loop := 1; loop <= maxResponsesFileSearchLoops; loop++ {
		query := responsesExtractQuery(&currentCall)
		if query == "" {
			break
		}

		// Embed the query.
		queryVec, err := w.embedder.EmbedText(query)
		if err != nil {
			break
		}

		// Search the vector store.
		maxR := w.maxResults
		if maxR <= 0 {
			maxR = 20
		}
		chunks, err := w.searcher.SearchChunks(w.vsIDs, queryVec, maxR)
		if err != nil {
			break
		}

		fsItemID := fmt.Sprintf("fs_%d_%d", loop, len(query))

		// Emit file_search_call events (streaming only; non-streaming builds them at end).
		if stream {
			events := w.inner.converter.FileSearchCallEvents(fsItemID, query, any(chunks))
			for _, ev := range events {
				if err := w.inner.writeEvent(ev.Event, ev.Data); err != nil {
					return 0, err
				}
			}
		}

		// Build follow-up messages: assistant tool call + tool result.
		assistantMsg := api.Message{
			Role:      "assistant",
			ToolCalls: []api.ToolCall{currentCall},
		}
		if currentResp.Message.Content != "" {
			assistantMsg.Content = currentResp.Message.Content
		}
		toolResultMsg := api.Message{
			Role:       "tool",
			Content:    responsesFormatFileSearchResults(chunks),
			ToolCallID: currentCall.ID,
		}
		followUpMessages = append(followUpMessages, assistantMsg, toolResultMsg)

		followUpResp, err := responsesCallFollowUp(w.chatReq.Model, followUpMessages, followUpTools, w.chatReq.Options)
		if err != nil {
			break
		}

		nextCall, hasMore := findResponsesFileSearchCall(followUpResp.Message.ToolCalls)
		if !hasMore {
			if stream {
				finalData, err := json.Marshal(followUpResp)
				if err != nil {
					return 0, err
				}
				return w.inner.writeResponse(finalData)
			}
			finalData, err := json.Marshal(followUpResp)
			if err != nil {
				return 0, err
			}
			return w.inner.writeResponseWithFileSearch(finalData, fsItemID, query, chunks)
		}
		currentResp = followUpResp
		currentCall = nextCall
	}

	// Fallback: emit the last response as-is.
	fallbackData, err := json.Marshal(currentResp)
	if err != nil {
		return 0, err
	}
	return w.inner.writeResponse(fallbackData)
}

// findResponsesFileSearchCall finds the first file_search tool call in a list.
func findResponsesFileSearchCall(toolCalls []api.ToolCall) (api.ToolCall, bool) {
	for _, tc := range toolCalls {
		if tc.Function.Name == "file_search" {
			return tc, true
		}
	}
	return api.ToolCall{}, false
}

// responsesFormatFileSearchResults formats chunk results as a plain-text tool message.
func responsesFormatFileSearchResults(chunks []FileChunkResult) string {
	var sb strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&sb, "%d. [%s] (score: %.3f)\n", i+1, c.Filename, c.Score)
		text := c.Text
		runes := []rune(text)
		if len(runes) > 500 {
			text = string(runes[:500]) + "..."
		}
		fmt.Fprintf(&sb, "   %s\n\n", text)
	}
	return sb.String()
}
