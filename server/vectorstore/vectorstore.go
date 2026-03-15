// Package vectorstore implements a local vector store backed by SQLite.
// It provides the storage layer for the file_search built-in tool in the
// Responses API, mirroring OpenAI's hosted vector store behaviour.
//
// Architecture:
//
//	VectorStore (1) ──< VectorStoreFile (N) ──< VectorStoreChunk (N)
//
// Each file is split into overlapping text chunks. Each chunk is embedded via
// /api/embed and stored as a JSON-encoded []float32. At query time, the query
// string is embedded with the same model and the top-K chunks are returned by
// cosine similarity — no SQLite extension required.
package vectorstore

import (
	"crypto/rand"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ─── Public types ────────────────────────────────────────────────────────────

// ExpiresAfter mirrors the OpenAI vector store expiry policy.
type ExpiresAfter struct {
	// Anchor is the reference point for the TTL. Supported values:
	// "last_active_at" (default) and "created_at".
	Anchor string `json:"anchor"`
	// Days is the number of days after the anchor time before the store expires.
	// A value of 0 means no per-store expiry.
	Days int `json:"days"`
}

// VectorStore is the metadata record for a vector store.
type VectorStore struct {
	ID           string        `json:"id"`
	Object       string        `json:"object"` // always "vector_store"
	Name         string        `json:"name"`
	CreatedAt    int64         `json:"created_at"`
	LastActiveAt int64         `json:"last_active_at"`
	Status       string        `json:"status"` // "completed" | "in_progress" | "expired"
	FileCounts   FileCounts    `json:"file_counts"`
	// ExpiresAfter is the per-store expiry policy. Nil means no per-store expiry.
	ExpiresAfter *ExpiresAfter `json:"expires_after,omitempty"`
	// ExpiresAt is the Unix timestamp when the store will expire, or 0 if none.
	ExpiresAt    int64         `json:"expires_at,omitempty"`
	// Metadata is an arbitrary JSON object supplied by the caller.
	Metadata     map[string]any `json:"metadata"`
}

// FileCounts mirrors the OpenAI vector store file_counts object.
type FileCounts struct {
	InProgress int `json:"in_progress"`
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
	Cancelled  int `json:"cancelled"`
	Total      int `json:"total"`
}

// VectorStoreFile is the metadata record for a file inside a vector store.
type VectorStoreFile struct {
	ID           string `json:"id"`
	Object       string `json:"object"` // always "vector_store.file"
	VectorStoreID string `json:"vector_store_id"`
	Filename     string `json:"filename"`
	MimeType     string `json:"mime_type"`
	Bytes        int    `json:"bytes"`
	Status       string `json:"status"` // "completed" | "in_progress" | "failed"
	CreatedAt    int64  `json:"created_at"`
	// EmbedModel is the model used to embed this file's chunks.
	EmbedModel string `json:"embed_model"`
}

// ChunkResult is a single search result returned by Search.
type ChunkResult struct {
	FileID    string  `json:"file_id"`
	Filename  string  `json:"filename"`
	Text      string  `json:"text"`
	Score     float64 `json:"score"`     // cosine similarity [0,1]
	ChunkIndex int    `json:"chunk_index"`
}

// EmbedFunc is the callback used to embed a text string.
// It returns a float32 slice (the embedding vector) or an error.
type EmbedFunc func(text string) ([]float32, error)

// ─── Store ───────────────────────────────────────────────────────────────────

// Store is a thread-safe local vector store backed by a single SQLite database.
type Store struct {
	mu   sync.RWMutex
	conn *sql.DB
}

// Open opens (or creates) the vector store database at the given path.
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("vectorstore: mkdir: %w", err)
	}
	conn, err := sql.Open("sqlite3", path+"?_foreign_keys=on&_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("vectorstore: open: %w", err)
	}
	s := &Store{conn: conn}
	if err := s.init(); err != nil {
		conn.Close()
		return nil, err
	}
	return s, nil
}

// Close closes the underlying database connection.
func (s *Store) Close() error {
	_, _ = s.conn.Exec("PRAGMA wal_checkpoint(TRUNCATE);")
	return s.conn.Close()
}

// ─── Schema ──────────────────────────────────────────────────────────────────

func (s *Store) init() error {
	_, err := s.conn.Exec(`
			CREATE TABLE IF NOT EXISTS vector_stores (
				id                   TEXT PRIMARY KEY,
				name                 TEXT NOT NULL DEFAULT '',
				created_at           INTEGER NOT NULL,
				last_active_at       INTEGER NOT NULL DEFAULT 0,
				expires_after_days   INTEGER NOT NULL DEFAULT 0,
				expires_after_anchor TEXT NOT NULL DEFAULT 'last_active_at',
				metadata             TEXT NOT NULL DEFAULT '{}'
			);

		CREATE TABLE IF NOT EXISTS vs_files (
			id              TEXT PRIMARY KEY,
			vector_store_id TEXT NOT NULL REFERENCES vector_stores(id) ON DELETE CASCADE,
			filename        TEXT NOT NULL,
			mime_type       TEXT NOT NULL DEFAULT '',
			bytes           INTEGER NOT NULL DEFAULT 0,
			status          TEXT NOT NULL DEFAULT 'in_progress',
			embed_model     TEXT NOT NULL DEFAULT '',
			created_at      INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS vs_files_vsid ON vs_files(vector_store_id);

		CREATE TABLE IF NOT EXISTS vs_chunks (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			file_id         TEXT NOT NULL REFERENCES vs_files(id) ON DELETE CASCADE,
			chunk_index     INTEGER NOT NULL,
			chunk_text      TEXT NOT NULL,
			embedding       TEXT NOT NULL  -- JSON []float32
		);
		CREATE INDEX IF NOT EXISTS vs_chunks_fileid ON vs_chunks(file_id);
	`)
	return err
}

// ─── Vector Store CRUD ───────────────────────────────────────────────────────

// CreateVectorStore creates a new vector store with the given name, metadata,
// and optional per-store expiry policy.
func (s *Store) CreateVectorStore(name string, metadata map[string]any, expiresAfter *ExpiresAfter) (*VectorStore, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := "vs_" + newID(24)
	now := time.Now().Unix()
	metaJSON, _ := json.Marshal(metadata)

	days := 0
	anchor := "last_active_at"
	if expiresAfter != nil && expiresAfter.Days > 0 {
		days = expiresAfter.Days
		if expiresAfter.Anchor != "" {
			anchor = expiresAfter.Anchor
		}
	}

	_, err := s.conn.Exec(
		`INSERT INTO vector_stores(id, name, created_at, last_active_at, expires_after_days, expires_after_anchor, metadata)
		 VALUES (?,?,?,?,?,?,?)`,
		id, name, now, now, days, anchor, string(metaJSON),
	)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: create: %w", err)
	}
	vs := &VectorStore{
		ID:           id,
		Object:       "vector_store",
		Name:         name,
		CreatedAt:    now,
		LastActiveAt: now,
		Status:       "completed",
		Metadata:     metadata,
	}
	if days > 0 {
		vs.ExpiresAfter = &ExpiresAfter{Anchor: anchor, Days: days}
		anchorTime := now
		if anchor == "created_at" {
			anchorTime = now
		}
		vs.ExpiresAt = anchorTime + int64(days)*86400
	}
	return vs, nil
}

// GetVectorStore retrieves a vector store by ID.
func (s *Store) GetVectorStore(id string) (*VectorStore, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.getVectorStore(id)
}

func (s *Store) getVectorStore(id string) (*VectorStore, error) {
	var vs VectorStore
	var metaJSON string
	var expiresAfterDays int
	var expiresAfterAnchor string
	err := s.conn.QueryRow(
		`SELECT id, name, created_at, last_active_at, expires_after_days, expires_after_anchor, metadata
		 FROM vector_stores WHERE id=?`, id,
	).Scan(&vs.ID, &vs.Name, &vs.CreatedAt, &vs.LastActiveAt, &expiresAfterDays, &expiresAfterAnchor, &metaJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("vectorstore: not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("vectorstore: get: %w", err)
	}
	vs.Object = "vector_store"
	vs.Status = "completed"
	_ = json.Unmarshal([]byte(metaJSON), &vs.Metadata)
	vs.FileCounts = s.fileCounts(id)
	if expiresAfterDays > 0 {
		vs.ExpiresAfter = &ExpiresAfter{Anchor: expiresAfterAnchor, Days: expiresAfterDays}
		anchorTime := vs.LastActiveAt
		if expiresAfterAnchor == "created_at" {
			anchorTime = vs.CreatedAt
		}
		vs.ExpiresAt = anchorTime + int64(expiresAfterDays)*86400
		if vs.ExpiresAt <= time.Now().Unix() {
			vs.Status = "expired"
		}
	}
	return &vs, nil
}

// ListVectorStores returns all vector stores ordered by creation time descending.
func (s *Store) ListVectorStores() ([]*VectorStore, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.conn.Query(`SELECT id FROM vector_stores ORDER BY created_at DESC`)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: list: %w", err)
	}
	defer rows.Close()

	var stores []*VectorStore
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		vs, err := s.getVectorStore(id)
		if err != nil {
			return nil, err
		}
		stores = append(stores, vs)
	}
	return stores, rows.Err()
}

// DeleteVectorStore deletes a vector store and all its files and chunks.
func (s *Store) DeleteVectorStore(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.conn.Exec(`DELETE FROM vector_stores WHERE id=?`, id)
	if err != nil {
		return fmt.Errorf("vectorstore: delete: %w", err)
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return fmt.Errorf("vectorstore: not found: %s", id)
	}
	return nil
}

func (s *Store) fileCounts(vsID string) FileCounts {
	var fc FileCounts
	rows, err := s.conn.Query(`SELECT status FROM vs_files WHERE vector_store_id=?`, vsID)
	if err != nil {
		return fc
	}
	defer rows.Close()
	for rows.Next() {
		var status string
		_ = rows.Scan(&status)
		fc.Total++
		switch status {
		case "completed":
			fc.Completed++
		case "in_progress":
			fc.InProgress++
		case "failed":
			fc.Failed++
		case "cancelled":
			fc.Cancelled++
		}
	}
	return fc
}

// ─── File CRUD ───────────────────────────────────────────────────────────────

// AddFile chunks, embeds, and stores a file in the given vector store.
// content is the raw file bytes; mimeType is used for display only.
// embedFn is called once per chunk to obtain its embedding vector.
// embedModel is stored for reference (e.g. "nomic-embed-text").
func (s *Store) AddFile(vsID, filename, mimeType string, content []byte, embedModel string, embedFn EmbedFunc) (*VectorStoreFile, error) {
	// Verify the vector store exists (read lock).
	s.mu.RLock()
	_, err := s.getVectorStore(vsID)
	s.mu.RUnlock()
	if err != nil {
		return nil, err
	}

	fileID := "file-" + newID(24)
	now := time.Now().Unix()

	// Insert the file record as in_progress.
	s.mu.Lock()
	_, err = s.conn.Exec(
		`INSERT INTO vs_files(id, vector_store_id, filename, mime_type, bytes, status, embed_model, created_at)
		 VALUES (?,?,?,?,?,?,?,?)`,
		fileID, vsID, filename, mimeType, len(content), "in_progress", embedModel, now,
	)
	s.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("vectorstore: insert file: %w", err)
	}

	// Chunk and embed outside the write lock.
	chunks := chunkText(extractText(content, mimeType), chunkSize, chunkOverlap)
	type embeddedChunk struct {
		idx  int
		text string
		vec  []float32
	}
	var embedded []embeddedChunk
	for i, chunk := range chunks {
		vec, err := embedFn(chunk)
		if err != nil {
			// Mark file as failed and return.
			s.mu.Lock()
			_, _ = s.conn.Exec(`UPDATE vs_files SET status='failed' WHERE id=?`, fileID)
			s.mu.Unlock()
			return nil, fmt.Errorf("vectorstore: embed chunk %d: %w", i, err)
		}
		embedded = append(embedded, embeddedChunk{i, chunk, vec})
	}

	// Bulk-insert chunks.
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.conn.Begin()
	if err != nil {
		return nil, fmt.Errorf("vectorstore: begin tx: %w", err)
	}
	stmt, err := tx.Prepare(`INSERT INTO vs_chunks(file_id, chunk_index, chunk_text, embedding) VALUES (?,?,?,?)`)
	if err != nil {
		tx.Rollback()
		return nil, fmt.Errorf("vectorstore: prepare: %w", err)
	}
	defer stmt.Close()

	for _, ec := range embedded {
		vecJSON, _ := json.Marshal(ec.vec)
		if _, err := stmt.Exec(fileID, ec.idx, ec.text, string(vecJSON)); err != nil {
			tx.Rollback()
			return nil, fmt.Errorf("vectorstore: insert chunk: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("vectorstore: commit: %w", err)
	}

	// Mark file as completed.
	_, _ = s.conn.Exec(`UPDATE vs_files SET status='completed' WHERE id=?`, fileID)

	return &VectorStoreFile{
		ID:            fileID,
		Object:        "vector_store.file",
		VectorStoreID: vsID,
		Filename:      filename,
		MimeType:      mimeType,
		Bytes:         len(content),
		Status:        "completed",
		EmbedModel:    embedModel,
		CreatedAt:     now,
	}, nil
}

// GetFile retrieves a file record by ID.
func (s *Store) GetFile(fileID string) (*VectorStoreFile, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.getFile(fileID)
}

func (s *Store) getFile(fileID string) (*VectorStoreFile, error) {
	var f VectorStoreFile
	err := s.conn.QueryRow(
		`SELECT id, vector_store_id, filename, mime_type, bytes, status, embed_model, created_at
		 FROM vs_files WHERE id=?`, fileID,
	).Scan(&f.ID, &f.VectorStoreID, &f.Filename, &f.MimeType, &f.Bytes, &f.Status, &f.EmbedModel, &f.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("vectorstore: file not found: %s", fileID)
	}
	if err != nil {
		return nil, fmt.Errorf("vectorstore: get file: %w", err)
	}
	f.Object = "vector_store.file"
	return &f, nil
}

// ListFiles returns all files in a vector store ordered by creation time descending.
func (s *Store) ListFiles(vsID string) ([]*VectorStoreFile, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.conn.Query(
		`SELECT id FROM vs_files WHERE vector_store_id=? ORDER BY created_at DESC`, vsID,
	)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: list files: %w", err)
	}
	defer rows.Close()

	var files []*VectorStoreFile
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		f, err := s.getFile(id)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	return files, rows.Err()
}

// DeleteFile removes a file and all its chunks from a vector store.
func (s *Store) DeleteFile(vsID, fileID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.conn.Exec(
		`DELETE FROM vs_files WHERE id=? AND vector_store_id=?`, fileID, vsID,
	)
	if err != nil {
		return fmt.Errorf("vectorstore: delete file: %w", err)
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return fmt.Errorf("vectorstore: file not found: %s", fileID)
	}
	return nil
}

// ─── Search ──────────────────────────────────────────────────────────────────

// Search returns the top maxResults chunks from the given vector stores that
// are most similar to queryVec (cosine similarity). If vsIDs is empty, all
// vector stores are searched.
func (s *Store) Search(vsIDs []string, queryVec []float32, maxResults int) ([]ChunkResult, error) {
	if maxResults <= 0 {
		maxResults = 20
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Build the SQL query.
	var query string
	var args []any
	if len(vsIDs) == 0 {
		query = `SELECT c.file_id, f.filename, c.chunk_index, c.chunk_text, c.embedding
				 FROM vs_chunks c JOIN vs_files f ON c.file_id=f.id
				 WHERE f.status='completed'`
	} else {
		placeholders := strings.Repeat("?,", len(vsIDs))
		placeholders = placeholders[:len(placeholders)-1]
		query = fmt.Sprintf(
			`SELECT c.file_id, f.filename, c.chunk_index, c.chunk_text, c.embedding
			 FROM vs_chunks c JOIN vs_files f ON c.file_id=f.id
			 WHERE f.status='completed' AND f.vector_store_id IN (%s)`, placeholders,
		)
		for _, id := range vsIDs {
			args = append(args, id)
		}
	}

	rows, err := s.conn.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: search query: %w", err)
	}
	defer rows.Close()

	type scored struct {
		ChunkResult
		score float64
	}
	var results []scored

	for rows.Next() {
		var fileID, filename, chunkText, embJSON string
		var chunkIdx int
		if err := rows.Scan(&fileID, &filename, &chunkIdx, &chunkText, &embJSON); err != nil {
			return nil, err
		}
		var vec []float32
		if err := json.Unmarshal([]byte(embJSON), &vec); err != nil {
			continue
		}
		score := cosineSimilarity(queryVec, vec)
		results = append(results, scored{
			ChunkResult: ChunkResult{
				FileID:     fileID,
				Filename:   filename,
				Text:       chunkText,
				ChunkIndex: chunkIdx,
			},
			score: score,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Sort by score descending.
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	if len(results) > maxResults {
		results = results[:maxResults]
	}

	out := make([]ChunkResult, len(results))
	for i, r := range results {
		r.ChunkResult.Score = r.score
		out[i] = r.ChunkResult
	}
	// Update last_active_at for all searched stores (supports "last_active_at" TTL).
	// We release the read lock first to avoid a lock upgrade deadlock.
	s.mu.RUnlock()
	for _, id := range vsIDs {
		s.touchLastActive(id)
	}
	s.mu.RLock() // re-acquire so defer RUnlock() is balanced
	return out, nil
}

// ─── Text chunking ───────────────────────────────────────────────────────────

const (
	chunkSize    = 800  // characters per chunk
	chunkOverlap = 200  // overlap between consecutive chunks
)

// extractText converts raw file bytes to plain text for chunking.
// For text/* MIME types the bytes are used directly. For other types a
// best-effort UTF-8 conversion is attempted.
func extractText(content []byte, mimeType string) string {
	// For plain text types, use as-is.
	if strings.HasPrefix(mimeType, "text/") ||
		mimeType == "application/json" ||
		mimeType == "application/typescript" ||
		mimeType == "application/x-sh" {
		return string(content)
	}
	// For binary types (PDF, DOCX, etc.) we fall back to the raw bytes
	// interpreted as UTF-8, stripping non-printable characters.
	// A production implementation would use a proper extractor library.
	var sb strings.Builder
	for _, r := range string(content) {
		if r >= 32 || r == '\n' || r == '\t' {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// chunkText splits text into overlapping chunks of at most size characters.
func chunkText(text string, size, overlap int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	var chunks []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for start := 0; start < len(runes); start += step {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[start:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}

// ─── Math helpers ────────────────────────────────────────────────────────────

// cosineSimilarity returns the cosine similarity between two vectors.
// Returns 0 if either vector has zero magnitude.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, magA, magB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		magA += float64(a[i]) * float64(a[i])
		magB += float64(b[i]) * float64(b[i])
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// ─── ID generation ───────────────────────────────────────────────────────────

// newID returns a random hex string of the given length.
func newID(n int) string {
	b := make([]byte, n/2+1)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)[:n]
}

// ─── TTL eviction ────────────────────────────────────────────────────────────

// touchLastActive updates the last_active_at timestamp for a vector store.
// Called internally after a successful Search to support "last_active_at" TTL.
func (s *Store) touchLastActive(vsID string) {
	_, _ = s.conn.Exec(
		`UPDATE vector_stores SET last_active_at=? WHERE id=?`,
		time.Now().Unix(), vsID,
	)
}

// EvictOlderThan deletes all vector stores that should be expired:
//   - Stores with no per-store expiry whose created_at is older than cutoff.
//   - Stores with a per-store ExpiresAfter policy whose computed expiry has passed.
//
// Cascading foreign keys automatically remove all associated vs_files and
// vs_chunks rows. Returns the total number of stores deleted.
func (s *Store) EvictOlderThan(cutoff time.Time) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().Unix()
	// Delete stores with no per-store TTL that are older than the global cutoff.
	res1, err := s.conn.Exec(
		`DELETE FROM vector_stores WHERE expires_after_days = 0 AND created_at < ?`,
		cutoff.Unix(),
	)
	if err != nil {
		return 0, fmt.Errorf("vectorstore: evict global: %w", err)
	}
	n1, _ := res1.RowsAffected()
	// Delete stores with a per-store TTL whose expiry has passed.
	// expires_at = anchor_time + expires_after_days * 86400
	// anchor_time is last_active_at when anchor='last_active_at', else created_at.
	res2, err := s.conn.Exec(`
		DELETE FROM vector_stores
		WHERE expires_after_days > 0
		  AND (
		    (expires_after_anchor = 'last_active_at' AND last_active_at + expires_after_days * 86400 <= ?)
		    OR
		    (expires_after_anchor != 'last_active_at' AND created_at + expires_after_days * 86400 <= ?)
		  )`,
		now, now,
	)
	if err != nil {
		return n1, fmt.Errorf("vectorstore: evict per-store: %w", err)
	}
	n2, _ := res2.RowsAffected()
	return n1 + n2, nil
}

// StartEviction launches a background goroutine that calls EvictOlderThan
// every interval. It evicts stores older than ttl. The goroutine stops when
// the provided done channel is closed. A zero ttl disables eviction entirely.
func (s *Store) StartEviction(ttl time.Duration, interval time.Duration, done <-chan struct{}) {
	if ttl <= 0 {
		return
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case t := <-ticker.C:
				cutoff := t.Add(-ttl)
				if n, err := s.EvictOlderThan(cutoff); err != nil {
					// Log but do not crash; eviction is best-effort.
					_ = err
				} else if n > 0 {
					_ = n // caller can observe via metrics if desired
				}
			}
		}
	}()
}
