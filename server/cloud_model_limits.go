package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/modelref"
)

type cloudModelLimit struct {
	Context int
	Output  int
}

var cloudModelLimits = map[string]cloudModelLimit{
	"minimax-m2.5":        {Context: 204_800, Output: 128_000},
	"cogito-2.1:671b":     {Context: 163_840, Output: 65_536},
	"deepseek-v3.1:671b":  {Context: 163_840, Output: 163_840},
	"deepseek-v3.2":       {Context: 163_840, Output: 65_536},
	"glm-4.6":             {Context: 202_752, Output: 131_072},
	"glm-4.7":             {Context: 202_752, Output: 131_072},
	"glm-5":               {Context: 202_752, Output: 131_072},
	"gpt-oss:120b":        {Context: 131_072, Output: 131_072},
	"gpt-oss:20b":         {Context: 131_072, Output: 131_072},
	"kimi-k2:1t":          {Context: 262_144, Output: 262_144},
	"kimi-k2.5":           {Context: 262_144, Output: 262_144},
	"kimi-k2-thinking":    {Context: 262_144, Output: 262_144},
	"nemotron-3-nano:30b": {Context: 1_048_576, Output: 131_072},
	"qwen3-coder:480b":    {Context: 262_144, Output: 65_536},
	"qwen3-coder-next":    {Context: 262_144, Output: 32_768},
	"qwen3-next:80b":      {Context: 262_144, Output: 32_768},
	"qwen3.5":             {Context: 262_144, Output: 32_768},
}

func lookupCloudModelLimit(name string) (cloudModelLimit, bool) {
	base, stripped := modelref.StripCloudSourceTag(name)
	if stripped {
		if l, ok := cloudModelLimits[base]; ok {
			return l, true
		}
	}
	if l, ok := cloudModelLimits[name]; ok {
		return l, true
	}
	// Fallback: serve the default immediately and trigger a background fetch
	// for models not in the static map. This avoids blocking the hot path on
	// a local HTTP round-trip. The result is cached and used on subsequent
	// requests.
	return fetchCloudModelLimitAsync(name)
}

// cloudModelLimitEntry is a cache entry with a TTL and an in-flight guard.
type cloudModelLimitEntry struct {
	limit     cloudModelLimit
	fetchedAt time.Time
	fetching  bool // true while a background goroutine is in flight
}

const (
	cloudModelLimitCacheTTL  = 10 * time.Minute
	cloudModelLimitCacheMax  = 256
)

var (
	cloudModelLimitCacheMu sync.Mutex
	cloudModelLimitCache   = make(map[string]*cloudModelLimitEntry, 64)
)

// fetchCloudModelLimitAsync returns the cached limit immediately (or zero
// value if not yet known) and schedules a background goroutine to refresh
// the cache entry if it is absent, expired, or stale.
func fetchCloudModelLimitAsync(name string) (cloudModelLimit, bool) {
	cloudModelLimitCacheMu.Lock()
	entry, ok := cloudModelLimitCache[name]
	now := time.Now()

	if ok && !entry.fetching && now.Sub(entry.fetchedAt) < cloudModelLimitCacheTTL {
		// Cache hit and still fresh — return immediately.
		l := entry.limit
		cloudModelLimitCacheMu.Unlock()
		return l, l.Context > 0
	}

	if ok && entry.fetching {
		// A fetch is already in flight — return whatever we have.
		l := entry.limit
		cloudModelLimitCacheMu.Unlock()
		return l, l.Context > 0
	}

	// Evict oldest entries when the cache is full.
	if !ok && len(cloudModelLimitCache) >= cloudModelLimitCacheMax {
		evictOldestCloudModelLimitEntry()
	}

	// Mark as fetching so only one goroutine is in flight per model.
	if !ok {
		cloudModelLimitCache[name] = &cloudModelLimitEntry{fetching: true}
	} else {
		entry.fetching = true
	}
	cloudModelLimitCacheMu.Unlock()

	// Launch background fetch — caller gets the default (zero) limit this time.
	go func() {
		l := doFetchCloudModelLimitFromOllama(name)
		cloudModelLimitCacheMu.Lock()
		cloudModelLimitCache[name] = &cloudModelLimitEntry{
			limit:     l,
			fetchedAt: time.Now(),
			fetching:  false,
		}
		cloudModelLimitCacheMu.Unlock()
	}()

	// Return whatever was in the cache before the fetch (may be zero).
	if ok {
		return entry.limit, entry.limit.Context > 0
	}
	return cloudModelLimit{}, false
}

// evictOldestCloudModelLimitEntry removes the entry with the oldest fetchedAt
// timestamp. Must be called with cloudModelLimitCacheMu held.
func evictOldestCloudModelLimitEntry() {
	var oldest string
	var oldestTime time.Time
	for k, v := range cloudModelLimitCache {
		if oldest == "" || v.fetchedAt.Before(oldestTime) {
			oldest = k
			oldestTime = v.fetchedAt
		}
	}
	if oldest != "" {
		delete(cloudModelLimitCache, oldest)
	}
}

// doFetchCloudModelLimitFromOllama performs the actual synchronous HTTP call
// to /api/show. It is always called from a background goroutine.
func doFetchCloudModelLimitFromOllama(name string) cloudModelLimit {
	type showRequest struct {
		Model string `json:"model"`
	}
	type showResponse struct {
		ModelInfo map[string]any `json:"model_info"`
	}

	body, _ := json.Marshal(showRequest{Model: name})
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(
		envconfig.Host().String()+"/api/show",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil || resp.StatusCode != http.StatusOK {
		return cloudModelLimit{} // negative cache — will retry after TTL
	}
	defer resp.Body.Close()

	var show showResponse
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return cloudModelLimit{}
	}

	var limit cloudModelLimit
	if ctx, ok := show.ModelInfo["context_length"]; ok {
		if v, ok := ctx.(float64); ok {
			limit.Context = int(v)
		}
	}
	if out, ok := show.ModelInfo["max_output_tokens"]; ok {
		if v, ok := out.(float64); ok {
			limit.Output = int(v)
		}
	}
	return limit
}
