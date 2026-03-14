package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"sync"

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
	// Fallback: query the local Ollama /api/show endpoint for models not in
	// the static map. This keeps limits accurate for newly deployed models
	// without requiring a fork update. Results are cached in-process.
	if l, ok := fetchCloudModelLimitFromOllama(name); ok {
		return l, true
	}
	return cloudModelLimit{}, false
}

// fetchCloudModelLimitFromOllama queries the local Ollama /api/show endpoint
// to retrieve context and output window sizes for a model not in the static
// cloudModelLimits map. Results are cached in-process to avoid repeated calls.
func fetchCloudModelLimitFromOllama(name string) (cloudModelLimit, bool) {
	cloudModelLimitCacheMu.Lock()
	if l, ok := cloudModelLimitCache[name]; ok {
		cloudModelLimitCacheMu.Unlock()
		return l, l.Context > 0
	}
	cloudModelLimitCacheMu.Unlock()

	type showRequest struct {
		Model string `json:"model"`
	}
	type showResponse struct {
		ModelInfo map[string]any `json:"model_info"`
	}

	body, _ := json.Marshal(showRequest{Model: name})
	resp, err := http.Post(
		envconfig.Host().String()+"/api/show",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil || resp.StatusCode != http.StatusOK {
		cloudModelLimitCacheMu.Lock()
		cloudModelLimitCache[name] = cloudModelLimit{} // negative cache
		cloudModelLimitCacheMu.Unlock()
		return cloudModelLimit{}, false
	}
	defer resp.Body.Close()

	var show showResponse
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return cloudModelLimit{}, false
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

	cloudModelLimitCacheMu.Lock()
	cloudModelLimitCache[name] = limit
	cloudModelLimitCacheMu.Unlock()

	return limit, limit.Context > 0
}

var (
	cloudModelLimitCacheMu sync.Mutex
	cloudModelLimitCache   = make(map[string]cloudModelLimit)
)
