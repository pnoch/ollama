package vectorstore_test

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/vectorstore"
)

// stubEmbed returns a deterministic fake embedding for a string by hashing it
// into a small fixed-dimension vector. This lets tests run without a real
// embedding model.
func stubEmbed(dims int) vectorstore.EmbedFunc {
	return func(text string) ([]float32, error) {
		v := make([]float32, dims)
		for i, r := range text {
			v[i%dims] += float32(r)
		}
		// Normalise so cosine similarity is well-defined.
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		norm := float32(math.Sqrt(sum))
		if norm > 0 {
			for i := range v {
				v[i] /= norm
			}
		}
		return v, nil
	}
}

func openTestStore(t *testing.T) *vectorstore.Store {
	t.Helper()
	dir := t.TempDir()
	s, err := vectorstore.Open(filepath.Join(dir, "test.db"))
	if err != nil {
		// go-sqlite3 requires CGO; skip gracefully when CGO is unavailable.
		if strings.Contains(err.Error(), "CGO_ENABLED=0") || strings.Contains(err.Error(), "cgo") {
			t.Skipf("skipping: sqlite3 requires CGO: %v", err)
		}
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// ─── VectorStore CRUD ─────────────────────────────────────────────────────────

func TestCreateAndGetVectorStore(t *testing.T) {
	s := openTestStore(t)

	vs, err := s.CreateVectorStore("my-store", map[string]any{"env": "test"})
	if err != nil {
		t.Fatalf("CreateVectorStore: %v", err)
	}
	if vs.Name != "my-store" {
		t.Errorf("Name = %q, want %q", vs.Name, "my-store")
	}
	if vs.ID == "" {
		t.Error("ID should not be empty")
	}

	got, err := s.GetVectorStore(vs.ID)
	if err != nil {
		t.Fatalf("GetVectorStore: %v", err)
	}
	if got.ID != vs.ID {
		t.Errorf("ID mismatch: got %q, want %q", got.ID, vs.ID)
	}
}

func TestListVectorStores(t *testing.T) {
	s := openTestStore(t)

	for _, name := range []string{"alpha", "beta", "gamma"} {
		if _, err := s.CreateVectorStore(name, nil); err != nil {
			t.Fatalf("CreateVectorStore(%q): %v", name, err)
		}
	}

	stores, err := s.ListVectorStores()
	if err != nil {
		t.Fatalf("ListVectorStores: %v", err)
	}
	if len(stores) != 3 {
		t.Errorf("len = %d, want 3", len(stores))
	}
}

func TestDeleteVectorStore(t *testing.T) {
	s := openTestStore(t)

	vs, _ := s.CreateVectorStore("to-delete", nil)
	if err := s.DeleteVectorStore(vs.ID); err != nil {
		t.Fatalf("DeleteVectorStore: %v", err)
	}

	if _, err := s.GetVectorStore(vs.ID); err == nil {
		t.Error("expected error after deletion, got nil")
	}
}

// ─── File upload and chunking ─────────────────────────────────────────────────

func TestAddAndListFiles(t *testing.T) {
	s := openTestStore(t)
	embed := stubEmbed(16)

	vs, _ := s.CreateVectorStore("docs", nil)

	content := []byte("Hello world. This is a test document. It has multiple sentences.")
	f, err := s.AddFile(vs.ID, "hello.txt", "text/plain", content, "stub", embed)
	if err != nil {
		t.Fatalf("AddFile: %v", err)
	}
	if f.Filename != "hello.txt" {
		t.Errorf("Filename = %q, want %q", f.Filename, "hello.txt")
	}
	if f.Status != "completed" {
		t.Errorf("Status = %q, want %q", f.Status, "completed")
	}

	files, err := s.ListFiles(vs.ID)
	if err != nil {
		t.Fatalf("ListFiles: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("len(files) = %d, want 1", len(files))
	}
}

func TestDeleteFile(t *testing.T) {
	s := openTestStore(t)
	embed := stubEmbed(16)

	vs, _ := s.CreateVectorStore("docs", nil)
	f, _ := s.AddFile(vs.ID, "a.txt", "text/plain", []byte("some text"), "stub", embed)

	if err := s.DeleteFile(vs.ID, f.ID); err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}

	files, _ := s.ListFiles(vs.ID)
	if len(files) != 0 {
		t.Errorf("expected 0 files after deletion, got %d", len(files))
	}
}

// ─── Search ───────────────────────────────────────────────────────────────────

func TestSearchReturnsRelevantChunks(t *testing.T) {
	s := openTestStore(t)
	embed := stubEmbed(32)

	vs, _ := s.CreateVectorStore("kb", nil)

	docs := []struct {
		name    string
		content string
	}{
		{"go.txt", "Go is a statically typed compiled programming language."},
		{"python.txt", "Python is a dynamically typed interpreted scripting language."},
		{"rust.txt", "Rust is a systems programming language focused on safety."},
	}
	for _, d := range docs {
		if _, err := s.AddFile(vs.ID, d.name, "text/plain", []byte(d.content), "stub", embed); err != nil {
			t.Fatalf("AddFile(%q): %v", d.name, err)
		}
	}

	// Query for "compiled language" — should rank go.txt highest.
	queryVec, _ := embed("compiled language")
	results, err := s.Search([]string{vs.ID}, queryVec, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	// Scores should be in descending order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: results[%d].Score=%f > results[%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func TestSearchAcrossMultipleVectorStores(t *testing.T) {
	s := openTestStore(t)
	embed := stubEmbed(16)

	vs1, _ := s.CreateVectorStore("store1", nil)
	vs2, _ := s.CreateVectorStore("store2", nil)

	s.AddFile(vs1.ID, "a.txt", "text/plain", []byte("cats and dogs"), "stub", embed)
	s.AddFile(vs2.ID, "b.txt", "text/plain", []byte("birds and fish"), "stub", embed)

	queryVec, _ := embed("animals")
	results, err := s.Search([]string{vs1.ID, vs2.ID}, queryVec, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) < 2 {
		t.Errorf("expected results from both stores, got %d", len(results))
	}
}

func TestSearchEmptyStore(t *testing.T) {
	s := openTestStore(t)
	embed := stubEmbed(16)

	vs, _ := s.CreateVectorStore("empty", nil)
	queryVec, _ := embed("anything")
	results, err := s.Search([]string{vs.ID}, queryVec, 5)
	if err != nil {
		t.Fatalf("Search on empty store: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

// ─── Chunking helpers ─────────────────────────────────────────────────────────

func TestChunkTextSplitsLongContent(t *testing.T) {
	// Build a string longer than the default chunk size.
	long := make([]byte, 3000)
	for i := range long {
		long[i] = 'a' + byte(i%26)
	}

	s := openTestStore(t)
	embed := stubEmbed(8)
	vs, _ := s.CreateVectorStore("big", nil)

	_, err := s.AddFile(vs.ID, "big.txt", "text/plain", long, "stub", embed)
	if err != nil {
		t.Fatalf("AddFile large file: %v", err)
	}
	// Verify multiple chunks were created by searching and getting multiple results.
	queryVec, _ := embed("aaabbbccc")
	results, err := s.Search([]string{vs.ID}, queryVec, 20)
	if err != nil {
		t.Fatalf("Search after large file: %v", err)
	}
	if len(results) < 2 {
		t.Errorf("expected multiple chunks for 3000-byte file, got %d results", len(results))
	}
}

// ─── Persistence ─────────────────────────────────────────────────────────────

func TestPersistenceAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "persist.db")
	embed := stubEmbed(16)

	// Create store and add a file.
	s1, err := vectorstore.Open(dbPath)
	if err != nil {
		if strings.Contains(err.Error(), "CGO_ENABLED=0") || strings.Contains(err.Error(), "cgo") {
			t.Skipf("skipping: sqlite3 requires CGO: %v", err)
		}
		t.Fatalf("Open (first): %v", err)
	}
	vs, _ := s1.CreateVectorStore("persist-test", nil)
	s1.AddFile(vs.ID, "doc.txt", "text/plain", []byte("persistent content"), "stub", embed)
	s1.Close()

	// Reopen and verify data is still there.
	s2, err := vectorstore.Open(dbPath)
	if err != nil {
		t.Fatalf("Open (second): %v", err)
	}
	defer s2.Close()

	files, err := s2.ListFiles(vs.ID)
	if err != nil {
		t.Fatalf("ListFiles after reopen: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file after reopen, got %d", len(files))
	}
	if files[0].Filename != "doc.txt" {
		t.Errorf("Filename = %q, want %q", files[0].Filename, "doc.txt")
	}

	// Verify the DB file actually exists on disk.
	if _, err := os.Stat(dbPath); err != nil {
		t.Errorf("DB file not found: %v", err)
	}
}

// ─── TTL eviction ─────────────────────────────────────────────────────────────

func TestEvictOlderThan(t *testing.T) {
	s := openTestStore(t)

	// Create two vector stores.
	old, err := s.CreateVectorStore("old-store", nil)
	if err != nil {
		t.Fatalf("CreateVectorStore old: %v", err)
	}
	recent, err := s.CreateVectorStore("recent-store", nil)
	if err != nil {
		t.Fatalf("CreateVectorStore recent: %v", err)
	}

	// Evict with a cutoff strictly before both stores were created — nothing deleted.
	// old.CreatedAt is a Unix timestamp (int64); subtract 1 second.
	cutoffBefore := time.Unix(old.CreatedAt-1, 0)
	n, err := s.EvictOlderThan(cutoffBefore)
	if err != nil {
		t.Fatalf("EvictOlderThan (before): %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 evictions with past cutoff, got %d", n)
	}

	// Evict with a cutoff 1 second after the most recent store — both deleted.
	cutoffAfter := time.Unix(recent.CreatedAt+1, 0)
	n, err = s.EvictOlderThan(cutoffAfter)
	if err != nil {
		t.Fatalf("EvictOlderThan (after): %v", err)
	}
	if n != 2 {
		t.Errorf("expected 2 evictions, got %d", n)
	}

	// Confirm both stores are gone.
	stores, err := s.ListVectorStores()
	if err != nil {
		t.Fatalf("ListVectorStores: %v", err)
	}
	if len(stores) != 0 {
		t.Errorf("expected 0 stores after eviction, got %d", len(stores))
	}
}
