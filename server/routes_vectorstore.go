package server

// routes_vectorstore.go — REST API handlers for /v1/vector_stores.
//
// These handlers mirror the OpenAI Vector Stores API so that clients
// (including Codex) can use the same SDK calls against a local ollama server.
//
// Endpoints implemented:
//   POST   /v1/vector_stores
//   GET    /v1/vector_stores
//   GET    /v1/vector_stores/:id
//   DELETE /v1/vector_stores/:id
//   POST   /v1/vector_stores/:id/files          (multipart or raw body)
//   GET    /v1/vector_stores/:id/files
//   GET    /v1/vector_stores/:id/files/:file_id
//   DELETE /v1/vector_stores/:id/files/:file_id

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/vectorstore"
)

// ─── Create vector store ─────────────────────────────────────────────────────

type createVectorStoreRequest struct {
	Name     string         `json:"name"`
	Metadata map[string]any `json:"metadata"`
}

func (s *Server) CreateVectorStoreHandler(c *gin.Context) {
	var req createVectorStoreRequest
	if err := c.ShouldBindJSON(&req); err != nil && err != io.EOF {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	vs, err := s.vectorStore.CreateVectorStore(req.Name, req.Metadata)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, vs)
}

// ─── List vector stores ───────────────────────────────────────────────────────

func (s *Server) ListVectorStoresHandler(c *gin.Context) {
	stores, err := s.vectorStore.ListVectorStores()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if stores == nil {
		stores = []*vectorstore.VectorStore{}
	}
	c.JSON(http.StatusOK, gin.H{
		"object":  "list",
		"data":    stores,
		"has_more": false,
	})
}

// ─── Get vector store ─────────────────────────────────────────────────────────

func (s *Server) GetVectorStoreHandler(c *gin.Context) {
	vs, err := s.vectorStore.GetVectorStore(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, vs)
}

// ─── Delete vector store ──────────────────────────────────────────────────────

func (s *Server) DeleteVectorStoreHandler(c *gin.Context) {
	id := c.Param("id")
	if err := s.vectorStore.DeleteVectorStore(id); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"object":  "vector_store.deleted",
		"deleted": true,
	})
}

// ─── Upload file to vector store ─────────────────────────────────────────────

// uploadFileRequest is the JSON body for non-multipart uploads.
type uploadFileRequest struct {
	// EmbedModel is the ollama model to use for embedding (e.g. "nomic-embed-text").
	// Defaults to the first available embedding model if omitted.
	EmbedModel string `json:"embed_model"`
}

func (s *Server) UploadVectorStoreFileHandler(c *gin.Context) {
	vsID := c.Param("id")

	// Verify the vector store exists before doing any work.
	if _, err := s.vectorStore.GetVectorStore(vsID); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	contentType := c.GetHeader("Content-Type")
	mediaType, params, _ := mime.ParseMediaType(contentType)

	var (
		filename   string
		mimeType   string
		content    []byte
		embedModel string
	)

	switch {
	case strings.HasPrefix(mediaType, "multipart/"):
		// Standard OpenAI SDK upload: multipart/form-data with "file" and
		// optional "embed_model" fields.
		mr := multipart.NewReader(c.Request.Body, params["boundary"])
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "malformed multipart: " + err.Error()})
				return
			}
			switch part.FormName() {
			case "file":
				filename = part.FileName()
				mimeType = part.Header.Get("Content-Type")
				if mimeType == "" {
					mimeType = "application/octet-stream"
				}
				var buf bytes.Buffer
				if _, err := io.Copy(&buf, io.LimitReader(part, 512<<20)); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "read file: " + err.Error()})
					return
				}
				content = buf.Bytes()
			case "embed_model":
				b, _ := io.ReadAll(part)
				embedModel = strings.TrimSpace(string(b))
			}
		}
	default:
		// Raw body upload: Content-Type is the file MIME type, filename from
		// X-Filename header or Content-Disposition.
		filename = c.GetHeader("X-Filename")
		if filename == "" {
			if _, p, err := mime.ParseMediaType(c.GetHeader("Content-Disposition")); err == nil {
				filename = p["filename"]
			}
		}
		if filename == "" {
			filename = "upload"
		}
		mimeType = mediaType
		if mimeType == "" {
			mimeType = "application/octet-stream"
		}
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, io.LimitReader(c.Request.Body, 512<<20)); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
			return
		}
		content = buf.Bytes()
		embedModel = c.Query("embed_model")
	}

	if len(content) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file content is empty"})
		return
	}

	// Infer MIME type from filename extension if not provided.
	if mimeType == "application/octet-stream" && filename != "" {
		if t := mime.TypeByExtension(filepath.Ext(filename)); t != "" {
			mimeType = t
		}
	}

	// Default embed model.
	if embedModel == "" {
		embedModel = "nomic-embed-text"
	}

	// Build the embed function that calls /api/embed on this server.
	embedFn := s.makeEmbedFunc(c.Request.Context(), embedModel)

	vsFile, err := s.vectorStore.AddFile(vsID, filename, mimeType, content, embedModel, embedFn)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, vsFile)
}

// makeEmbedFunc returns an EmbedFunc that calls /api/embed on this server.
func (s *Server) makeEmbedFunc(ctx context.Context, model string) vectorstore.EmbedFunc {
	return func(text string) ([]float32, error) {
		client := api.NewClient(nil, nil)
		resp, err := client.Embed(ctx, &api.EmbedRequest{
			Model: model,
			Input: text,
		})
		if err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}
		if len(resp.Embeddings) == 0 {
			return nil, fmt.Errorf("embed: no embeddings returned")
		}
		return resp.Embeddings[0], nil
	}
}

// ─── List files in vector store ───────────────────────────────────────────────

func (s *Server) ListVectorStoreFilesHandler(c *gin.Context) {
	vsID := c.Param("id")
	if _, err := s.vectorStore.GetVectorStore(vsID); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	files, err := s.vectorStore.ListFiles(vsID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if files == nil {
		files = []*vectorstore.VectorStoreFile{}
	}
	c.JSON(http.StatusOK, gin.H{
		"object":  "list",
		"data":    files,
		"has_more": false,
	})
}

// ─── Get file in vector store ─────────────────────────────────────────────────

func (s *Server) GetVectorStoreFileHandler(c *gin.Context) {
	file, err := s.vectorStore.GetFile(c.Param("file_id"))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, file)
}

// ─── Delete file from vector store ───────────────────────────────────────────

func (s *Server) DeleteVectorStoreFileHandler(c *gin.Context) {
	vsID := c.Param("id")
	fileID := c.Param("file_id")
	if err := s.vectorStore.DeleteFile(vsID, fileID); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"id":      fileID,
		"object":  "vector_store.file.deleted",
		"deleted": true,
	})
}

// ─── FileSearcher / FileEmbedder adapters ─────────────────────────────────────
//
// These adapters satisfy the middleware.FileSearcher and middleware.FileEmbedder
// interfaces without creating an import cycle (server → middleware → server).
// They are injected into the gin context by FileSearchDepsMiddleware so that
// ResponsesMiddleware can pick them up via c.Get("file_searcher") /
// c.Get("file_embedder").

// serverFileSearcher wraps *vectorstore.Store and converts its ChunkResult
// to the map[string]any shape that middleware.FileChunkResult expects.
type serverFileSearcher struct {
	store *vectorstore.Store
}

// SearchChunks implements middleware.FileSearcher.
// It returns []FileChunkResult as []any to satisfy the interface without
// importing the middleware package.
func (s *serverFileSearcher) SearchChunks(vsIDs []string, queryVec []float32, maxResults int) ([]any, error) {
	chunks, err := s.store.Search(vsIDs, queryVec, maxResults)
	if err != nil {
		return nil, err
	}
	out := make([]any, len(chunks))
	for i, c := range chunks {
		out[i] = map[string]any{
			"file_id":     c.FileID,
			"filename":    c.Filename,
			"text":        c.Text,
			"score":       c.Score,
			"chunk_index": c.ChunkIndex,
		}
	}
	return out, nil
}

// serverFileEmbedder wraps the server's makeEmbedFunc.
type serverFileEmbedder struct {
	embedFn func(string) ([]float32, error)
}

// EmbedText implements middleware.FileEmbedder.
func (e *serverFileEmbedder) EmbedText(text string) ([]float32, error) {
	return e.embedFn(text)
}

// FileSearchDepsMiddleware returns a gin middleware that injects the
// FileSearcher and FileEmbedder into the gin context so that
// ResponsesMiddleware can use them for file_search built-in tool calls.
//
// embedModel is the ollama embedding model to use (e.g. "nomic-embed-text").
// If empty, "nomic-embed-text" is used as the default.
func (s *Server) FileSearchDepsMiddleware(embedModel string) func(*gin.Context) {
	if embedModel == "" {
		embedModel = "nomic-embed-text"
	}
	searcher := &serverFileSearcher{store: s.vectorStore}
	return func(c *gin.Context) {
		embedder := &serverFileEmbedder{
			embedFn: s.makeEmbedFunc(c.Request.Context(), embedModel),
		}
		c.Set("file_searcher", searcher)
		c.Set("file_embedder", embedder)
		c.Next()
	}
}
