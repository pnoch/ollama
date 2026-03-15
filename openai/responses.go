package openai

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// ResponsesContent is a discriminated union for input content types.
// Concrete types: ResponsesTextContent, ResponsesImageContent
type ResponsesContent interface {
	responsesContent() // unexported marker method
}

type ResponsesTextContent struct {
	Type string `json:"type"` // always "input_text"
	Text string `json:"text"`
}

func (ResponsesTextContent) responsesContent() {}

type ResponsesImageContent struct {
	Type string `json:"type"` // always "input_image"
	// TODO(drifkin): is this really required? that seems verbose and a default is specified in the docs
	Detail   string `json:"detail"`              // required
	FileID   string `json:"file_id,omitempty"`   // optional
	ImageURL string `json:"image_url,omitempty"` // optional
}

func (ResponsesImageContent) responsesContent() {}

// ResponsesOutputTextContent represents output text from a previous assistant response
// that is being passed back as part of the conversation history.
type ResponsesOutputTextContent struct {
	Type string `json:"type"` // always "output_text"
	Text string `json:"text"`
}

func (ResponsesOutputTextContent) responsesContent() {}

type ResponsesInputMessage struct {
	Type    string             `json:"type"` // always "message"
	Role    string             `json:"role"` // one of `user`, `system`, `developer`
	Content []ResponsesContent `json:"content,omitempty"`
}

func (m *ResponsesInputMessage) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	m.Type = aux.Type
	m.Role = aux.Role

	if len(aux.Content) == 0 {
		return nil
	}

	// Try to parse content as a string first (shorthand format)
	var contentStr string
	if err := json.Unmarshal(aux.Content, &contentStr); err == nil {
		m.Content = []ResponsesContent{
			ResponsesTextContent{Type: "input_text", Text: contentStr},
		}
		return nil
	}

	// Otherwise, parse as an array of content items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Content, &rawItems); err != nil {
		return fmt.Errorf("content must be a string or array: %w", err)
	}

	m.Content = make([]ResponsesContent, 0, len(rawItems))
	for i, raw := range rawItems {
		// Peek at the type field to determine which concrete type to use
		var typeField struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &typeField); err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}

		switch typeField.Type {
		case "input_text":
			var content ResponsesTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "input_image":
			var content ResponsesImageContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "output_text":
			var content ResponsesOutputTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		default:
			return fmt.Errorf("content[%d]: unknown content type: %s", i, typeField.Type)
		}
	}

	return nil
}

type ResponsesOutputMessage struct{}

// ResponsesInputItem is a discriminated union for input items.
// Concrete types: ResponsesInputMessage (more to come)
type ResponsesInputItem interface {
	responsesInputItem() // unexported marker method
}

func (ResponsesInputMessage) responsesInputItem() {}

// ResponsesFunctionCall represents an assistant's function call in conversation history.
type ResponsesFunctionCall struct {
	ID        string `json:"id,omitempty"` // item ID
	Type      string `json:"type"`         // always "function_call"
	CallID    string `json:"call_id"`      // the tool call ID
	Name      string `json:"name"`         // function name
	Arguments string `json:"arguments"`    // JSON arguments string
}

func (ResponsesFunctionCall) responsesInputItem() {}

// ResponsesFunctionCallOutput represents a function call result from the client.
type ResponsesFunctionCallOutput struct {
	Type   string `json:"type"`    // always "function_call_output"
	CallID string `json:"call_id"` // links to the original function call
	Output string `json:"output"`  // the function result
}

func (ResponsesFunctionCallOutput) responsesInputItem() {}

// ResponsesCustomToolCall represents a freeform/custom tool call in conversation history.
type ResponsesCustomToolCall struct {
	ID     string `json:"id,omitempty"`
	Type   string `json:"type"` // always "custom_tool_call"
	Status string `json:"status,omitempty"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	Input  string `json:"input"`
}

func (ResponsesCustomToolCall) responsesInputItem() {}

// ResponsesCustomToolCallOutput represents a custom tool call result from the client.
type ResponsesCustomToolCallOutput struct {
	Type   string `json:"type"`    // always "custom_tool_call_output"
	CallID string `json:"call_id"` // links to the original custom tool call
	Output string `json:"output"`  // the tool result
}

func (ResponsesCustomToolCallOutput) responsesInputItem() {}

// ResponsesReasoningInput represents a reasoning item passed back as input.
// This is used when the client sends previous reasoning back for context.
type ResponsesReasoningInput struct {
	ID               string                      `json:"id,omitempty"`
	Type             string                      `json:"type"` // always "reasoning"
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent string                      `json:"encrypted_content,omitempty"`
}

func (ResponsesReasoningInput) responsesInputItem() {}

// ResponsesWebSearchCall represents a web search call in conversation history.
type ResponsesWebSearchCall struct {
	Type   string `json:"type"` // always "web_search_call"
	Status string `json:"status,omitempty"`
	Action struct {
		Type  string `json:"type"`
		Query string `json:"query,omitempty"`
	} `json:"action"`
}

func (ResponsesWebSearchCall) responsesInputItem() {}

// ResponsesImageGenerationCall represents an image generation call in conversation history.
type ResponsesImageGenerationCall struct {
	Type   string `json:"type"` // always "image_generation_call"
	Status string `json:"status,omitempty"`
	Action struct {
		Type   string `json:"type"`
		Prompt string `json:"prompt,omitempty"`
	} `json:"action"`
}

func (ResponsesImageGenerationCall) responsesInputItem() {}

// unmarshalResponsesInputItem unmarshals a single input item from JSON.
func unmarshalResponsesInputItem(data []byte) (ResponsesInputItem, error) {
	var typeField struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	// Handle shorthand message format: {"role": "...", "content": "..."}
	// When type is empty but role is present, treat as a message
	itemType := typeField.Type
	if itemType == "" && typeField.Role != "" {
		itemType = "message"
	}

	switch itemType {
	case "message":
		var msg ResponsesInputMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, err
		}
		return msg, nil
	case "function_call":
		var fc ResponsesFunctionCall
		if err := json.Unmarshal(data, &fc); err != nil {
			return nil, err
		}
		return fc, nil
	case "function_call_output":
		var output ResponsesFunctionCallOutput
		if err := json.Unmarshal(data, &output); err != nil {
			return nil, err
		}
		return output, nil
	case "custom_tool_call":
		var custom ResponsesCustomToolCall
		if err := json.Unmarshal(data, &custom); err != nil {
			return nil, err
		}
		return custom, nil
	case "custom_tool_call_output":
		var output ResponsesCustomToolCallOutput
		if err := json.Unmarshal(data, &output); err != nil {
			return nil, err
		}
		return output, nil
	case "reasoning":
		var reasoning ResponsesReasoningInput
		if err := json.Unmarshal(data, &reasoning); err != nil {
			return nil, err
		}
		return reasoning, nil
	case "web_search_call":
		var ws ResponsesWebSearchCall
		if err := json.Unmarshal(data, &ws); err != nil {
			return nil, err
		}
		return ws, nil
	case "image_generation_call":
		var ig ResponsesImageGenerationCall
		if err := json.Unmarshal(data, &ig); err != nil {
			return nil, err
		}
		return ig, nil
	default:
		return nil, fmt.Errorf("unknown input item type: %s", typeField.Type)
	}
}

// ResponsesInput can be either:
// - a string (equivalent to a text input with the user role)
// - an array of input items (see ResponsesInputItem)
type ResponsesInput struct {
	Text  string               // set if input was a plain string
	Items []ResponsesInputItem // set if input was an array
}

func (r *ResponsesInput) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		r.Text = s
		return nil
	}

	// Otherwise, try array of input items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(data, &rawItems); err != nil {
		return fmt.Errorf("input must be a string or array: %w", err)
	}

	r.Items = make([]ResponsesInputItem, 0, len(rawItems))
	for i, raw := range rawItems {
		item, err := unmarshalResponsesInputItem(raw)
		if err != nil {
			return fmt.Errorf("input[%d]: %w", i, err)
		}
		r.Items = append(r.Items, item)
	}

	return nil
}

type ResponsesReasoning struct {
	// originally: optional, default is per-model
	Effort string `json:"effort,omitempty"`

	// originally: deprecated, use `summary` instead. One of `auto`, `concise`, `detailed`
	GenerateSummary string `json:"generate_summary,omitempty"`

	// originally: optional, one of `auto`, `concise`, `detailed`
	Summary string `json:"summary,omitempty"`
}

type ResponsesTextFormat struct {
	Type   string          `json:"type"`             // "text", "json_schema"
	Name   string          `json:"name,omitempty"`   // for json_schema
	Schema json.RawMessage `json:"schema,omitempty"` // for json_schema
	Strict *bool           `json:"strict,omitempty"` // for json_schema
}

type ResponsesText struct {
	Format *ResponsesTextFormat `json:"format,omitempty"`
}

// ResponsesTool represents a tool in the Responses API format.
// Note: This differs from api.Tool which nests fields under "function".
//
// The Type field discriminates between user-defined function tools and
// built-in hosted tools:
//   - "function"            — user-defined function (Name/Description/Parameters present)
//   - "web_search_preview"  — hosted web search (SearchContextSize/AllowedDomains may be set)
//   - "file_search"         — hosted vector-store search (VectorStoreIDs/MaxNumResults may be set)
//   - "computer_use_preview"— hosted computer control (not yet implemented locally)
type ResponsesTool struct {
	Type        string         `json:"type"` // "function", "web_search_preview", "file_search", ...
	Name        string         `json:"name"`
	Description *string        `json:"description"` // nullable but required
	Strict      *bool          `json:"strict"`      // nullable but required
	Parameters  map[string]any `json:"parameters"`  // nullable but required

	// web_search_preview fields
	SearchContextSize string   `json:"search_context_size,omitempty"` // "low", "medium", "high"
	AllowedDomains    []string `json:"allowed_domains,omitempty"`

	// file_search fields
	VectorStoreIDs []string `json:"vector_store_ids,omitempty"`
	MaxNumResults  int      `json:"max_num_results,omitempty"` // default 20
	// EmbedModel is an ollama-specific extension: the embedding model to use
	// when searching. Defaults to "nomic-embed-text" if omitted.
	EmbedModel string `json:"embed_model,omitempty"`
}

// HasWebSearchPreview reports whether the tool list contains a web_search_preview
// built-in tool, and returns the first one found.
func HasWebSearchPreview(tools []ResponsesTool) (ResponsesTool, bool) {
	for _, t := range tools {
		if strings.HasPrefix(t.Type, "web_search") {
			return t, true
		}
	}
	return ResponsesTool{}, false
}

// HasFileSearch reports whether the tool list contains a file_search built-in
// tool, and returns the first one found.
func HasFileSearch(tools []ResponsesTool) (ResponsesTool, bool) {
	for _, t := range tools {
		if t.Type == "file_search" {
			return t, true
		}
	}
	return ResponsesTool{}, false
}

type ResponsesRequest struct {
	Model string `json:"model"`

	// originally: optional, default is false
	// for us: not supported
	Background bool `json:"background"`

	// originally: optional `string | {id: string}`
	// for us: not supported
	Conversation json.RawMessage `json:"conversation"`

	// originally: string[]
	// for us: ignored
	Include []string `json:"include"`

	Input ResponsesInput `json:"input"`

	// optional, inserts a system message at the start of the conversation
	Instructions string `json:"instructions,omitempty"`

	// optional, maps to num_predict
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	Reasoning ResponsesReasoning `json:"reasoning"`

	// optional, default is 1.0
	Temperature *float64 `json:"temperature"`

	// optional, controls output format (e.g. json_schema)
	Text *ResponsesText `json:"text,omitempty"`

	// optional, default is 1.0
	TopP *float64 `json:"top_p"`

	// optional, default is `"disabled"`
	Truncation *string `json:"truncation"`

	Tools []ResponsesTool `json:"tools,omitempty"`

	// optional: "auto" | "none" | "required" | {type: "function", name: string}
	// Passed through to api.ChatRequest.ToolChoice; the core server applies
	// grammar-constrained generation (GBNF via llama.SchemaToGrammar) and a
	// belt-and-suspenders system message for "required" / named function choices.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`

	// optional, default is false
	Stream *bool `json:"stream,omitempty"`
}

// FromResponsesRequest converts a ResponsesRequest to api.ChatRequest
func FromResponsesRequest(r ResponsesRequest) (*api.ChatRequest, error) {
	var messages []api.Message

	// Add instructions as system message if present
	if r.Instructions != "" {
		messages = append(messages, api.Message{
			Role:    "system",
			Content: r.Instructions,
		})
	}

	// Handle simple string input
	if r.Input.Text != "" {
		messages = append(messages, api.Message{
			Role:    "user",
			Content: r.Input.Text,
		})
	}

	// Handle array of input items
	// Track pending reasoning to merge with the next assistant message
	var pendingThinking string

	for _, item := range r.Input.Items {
		switch v := item.(type) {
		case ResponsesReasoningInput:
			// Store thinking to merge with the next assistant message
			pendingThinking = v.EncryptedContent
		case ResponsesInputMessage:
			msg, err := convertInputMessage(v)
			if err != nil {
				return nil, err
			}
			// If this is an assistant message, attach pending thinking
			if msg.Role == "assistant" && pendingThinking != "" {
				msg.Thinking = pendingThinking
				pendingThinking = ""
			}
			messages = append(messages, msg)
		case ResponsesFunctionCall:
			// Convert function call to assistant message with tool calls
			var args api.ToolCallFunctionArguments
			if v.Arguments != "" {
				if err := json.Unmarshal([]byte(v.Arguments), &args); err != nil {
					return nil, fmt.Errorf("failed to parse function call arguments: %w", err)
				}
			}
			toolCall := api.ToolCall{
				ID: v.CallID,
				Function: api.ToolCallFunction{
					Name:      v.Name,
					Arguments: args,
				},
			}

			// Merge tool call into existing assistant message if it has content or tool calls
			if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
				lastMsg := &messages[len(messages)-1]
				lastMsg.ToolCalls = append(lastMsg.ToolCalls, toolCall)
				if pendingThinking != "" {
					lastMsg.Thinking = pendingThinking
					pendingThinking = ""
				}
			} else {
				msg := api.Message{
					Role:      "assistant",
					ToolCalls: []api.ToolCall{toolCall},
				}
				if pendingThinking != "" {
					msg.Thinking = pendingThinking
					pendingThinking = ""
				}
				messages = append(messages, msg)
			}
		case ResponsesFunctionCallOutput:
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    v.Output,
				ToolCallID: v.CallID,
			})
		case ResponsesCustomToolCall:
			args, err := parseToolCallArguments(v.Input)
			if err != nil {
				return nil, fmt.Errorf("failed to parse custom tool call input: %w", err)
			}
			toolCall := api.ToolCall{
				ID: v.CallID,
				Function: api.ToolCallFunction{
					Name:      v.Name,
					Arguments: args,
				},
			}

			if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
				lastMsg := &messages[len(messages)-1]
				lastMsg.ToolCalls = append(lastMsg.ToolCalls, toolCall)
				if pendingThinking != "" {
					lastMsg.Thinking = pendingThinking
					pendingThinking = ""
				}
			} else {
				msg := api.Message{
					Role:      "assistant",
					ToolCalls: []api.ToolCall{toolCall},
				}
				if pendingThinking != "" {
					msg.Thinking = pendingThinking
					pendingThinking = ""
				}
				messages = append(messages, msg)
			}
		case ResponsesCustomToolCallOutput:
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    v.Output,
				ToolCallID: v.CallID,
			})
		}
	}

	// If there's trailing reasoning without a following message, emit it
	if pendingThinking != "" {
		messages = append(messages, api.Message{
			Role:     "assistant",
			Thinking: pendingThinking,
		})
	}

	options := make(map[string]any)

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else {
		// Default to 1.0 per OpenAI spec. This matches the value already used
		// in the response object and avoids model-specific default drift.
		options["top_p"] = 1.0
	}

	if r.MaxOutputTokens != nil {
		options["num_predict"] = *r.MaxOutputTokens
	}

	// Convert tools from Responses API format to api.Tool format.
	// tool_choice is passed through to api.ChatRequest.ToolChoice so that the
	// core server can apply grammar-constrained generation (GBNF via
	// llama.SchemaToGrammar) and system-message enforcement at the sampler level.
	//
	// Built-in hosted tools (web_search_preview, file_search, computer_use_preview)
	// are translated to equivalent function tool definitions so that local models
	// can emit structured tool calls. The middleware layer (ResponsesWebSearchWriter)
	// intercepts those calls, executes them server-side, and feeds results back.
	var tools []api.Tool
	for _, t := range r.Tools {
		switch {
		case strings.HasPrefix(t.Type, "web_search"):
			// Translate web_search_preview → web_search function tool.
			// The ResponsesWebSearchWriter in middleware/openai.go intercepts
			// web_search tool calls and executes them server-side.
			tools = append(tools, webSearchFunctionTool())
		case t.Type == "file_search":
			// Translate file_search → file_search function tool.
			// The ResponsesFileSearchWriter in middleware/openai.go intercepts
			// file_search tool calls, queries the local vector store, and feeds
			// results back to the model.
			tools = append(tools, fileSearchFunctionTool())
		default:
			// function tools and unrecognised built-ins pass through as-is.
			tool, err := convertTool(t)
			if err != nil {
				return nil, err
			}
			tools = append(tools, tool)
		}
	}

	// Handle text format (e.g. json_schema)
	var format json.RawMessage
	if r.Text != nil && r.Text.Format != nil {
		switch r.Text.Format.Type {
		case "json_schema":
			if r.Text.Format.Schema != nil {
				format = r.Text.Format.Schema
			}
		}
	}

	return &api.ChatRequest{
		Model:      r.Model,
		Messages:   messages,
		Options:    options,
		Tools:      tools,
		Format:     format,
		ToolChoice: r.ToolChoice, // passed through for core grammar-constrained enforcement
	}, nil
}

func parseToolCallArguments(input string) (api.ToolCallFunctionArguments, error) {
	var args api.ToolCallFunctionArguments
	if strings.TrimSpace(input) != "" && json.Unmarshal([]byte(input), &args) == nil {
		return args, nil
	}

	args = api.NewToolCallFunctionArguments()
	args.Set("input", input)
	return args, nil
}

func convertTool(t ResponsesTool) (api.Tool, error) {
	// Convert parameters from map[string]any to api.ToolFunctionParameters
	var params api.ToolFunctionParameters
	if t.Parameters != nil {
		// Marshal and unmarshal to convert
		b, err := json.Marshal(t.Parameters)
		if err != nil {
			return api.Tool{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return api.Tool{}, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
	}

	var description string
	if t.Description != nil {
		description = *t.Description
	}

	return api.Tool{
		Type: t.Type,
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: description,
			Parameters:  params,
		},
	}, nil
}

// webSearchFunctionTool returns the api.Tool definition for the built-in web_search
// function. This is injected into the tool list when the client requests
// web_search_preview so that local models can emit structured web_search calls.
func webSearchFunctionTool() api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The search query to look up on the web",
	})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "web_search",
			Description: "Search the web for current information. Use this when you need up-to-date information that may not be in your training data.",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
				Required:   []string{"query"},
			},
		},
	}
}

// fileSearchFunctionTool returns the api.Tool definition for the built-in file_search
// function. This is injected into the tool list when the client requests
// file_search so that local models can emit structured file_search calls.
func fileSearchFunctionTool() api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The search query to find relevant information in uploaded files",
	})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "file_search",
			Description: "Search through uploaded files and documents to find relevant information. Use this when the user's question can be answered from their uploaded files.",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
				Required:   []string{"query"},
			},
		},
	}
}

func convertInputMessage(m ResponsesInputMessage) (api.Message, error) {
	var content string
	var images []api.ImageData

	for _, c := range m.Content {
		switch v := c.(type) {
		case ResponsesTextContent:
			content += v.Text
		case ResponsesOutputTextContent:
			content += v.Text
		case ResponsesImageContent:
			if v.ImageURL == "" {
				continue // Skip if no URL (FileID not supported)
			}
			img, err := decodeImageURL(v.ImageURL)
			if err != nil {
				return api.Message{}, err
			}
			images = append(images, img)
		}
	}

	return api.Message{
		Role:    m.Role,
		Content: content,
		Images:  images,
	}, nil
}

// Response types for the Responses API

// ResponsesTextField represents the text output configuration in the response.
type ResponsesTextField struct {
	Format ResponsesTextFormat `json:"format"`
}

// ResponsesReasoningOutput represents reasoning configuration in the response.
type ResponsesReasoningOutput struct {
	Effort  *string `json:"effort,omitempty"`
	Summary *string `json:"summary,omitempty"`
}

// ResponsesError represents an error in the response.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesIncompleteDetails represents details about why a response was incomplete.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

type ResponsesResponse struct {
	ID                 string                      `json:"id"`
	Object             string                      `json:"object"`
	CreatedAt          int64                       `json:"created_at"`
	CompletedAt        *int64                      `json:"completed_at"`
	Status             string                      `json:"status"`
	IncompleteDetails  *ResponsesIncompleteDetails `json:"incomplete_details"`
	Model              string                      `json:"model"`
	PreviousResponseID *string                     `json:"previous_response_id"`
	Instructions       *string                     `json:"instructions"`
	Output             []ResponsesOutputItem       `json:"output"`
	Error              *ResponsesError             `json:"error"`
	Tools              []ResponsesTool             `json:"tools"`
	ToolChoice         any                         `json:"tool_choice"`
	Truncation         string                      `json:"truncation"`
	ParallelToolCalls  bool                        `json:"parallel_tool_calls"`
	Text               ResponsesTextField          `json:"text"`
	TopP               float64                     `json:"top_p"`
	PresencePenalty    float64                     `json:"presence_penalty"`
	FrequencyPenalty   float64                     `json:"frequency_penalty"`
	TopLogprobs        int                         `json:"top_logprobs"`
	Temperature        float64                     `json:"temperature"`
	Reasoning          *ResponsesReasoningOutput   `json:"reasoning"`
	Usage              *ResponsesUsage             `json:"usage"`
	MaxOutputTokens    *int                        `json:"max_output_tokens"`
	MaxToolCalls       *int                        `json:"max_tool_calls"`
	Store              bool                        `json:"store"`
	Background         bool                        `json:"background"`
	ServiceTier        string                      `json:"service_tier"`
	Metadata           map[string]any              `json:"metadata"`
	SafetyIdentifier   *string                     `json:"safety_identifier"`
	PromptCacheKey     *string                     `json:"prompt_cache_key"`
}

type ResponsesOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"` // "message", "function_call", or "reasoning"
	Status    string                   `json:"status,omitempty"`
	Role      string                   `json:"role,omitempty"`      // for message
	Content   []ResponsesOutputContent `json:"content,omitempty"`   // for message
	CallID    string                   `json:"call_id,omitempty"`   // for function_call
	Name      string                   `json:"name,omitempty"`      // for function_call
	Arguments string                   `json:"arguments,omitempty"` // for function_call

	// Reasoning fields
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`           // for reasoning
	EncryptedContent string                      `json:"encrypted_content,omitempty"` // for reasoning
}

type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

type ResponsesOutputContent struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
}

type ResponsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponsesUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	OutputTokens        int                          `json:"output_tokens"`
	TotalTokens         int                          `json:"total_tokens"`
	InputTokensDetails  ResponsesInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails ResponsesOutputTokensDetails `json:"output_tokens_details"`
}

// derefFloat64 returns the value of a float64 pointer, or a default if nil.
func derefFloat64(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// ToResponse converts an api.ChatResponse to a Responses API response.
// The request is used to echo back request parameters in the response.
func ToResponse(model, responseID, itemID string, chatResponse api.ChatResponse, request ResponsesRequest) ResponsesResponse {
	var output []ResponsesOutputItem

	// Add reasoning item if thinking is present
	if chatResponse.Message.Thinking != "" {
		output = append(output, ResponsesOutputItem{
			ID:   fmt.Sprintf("rs_%s", responseID),
			Type: "reasoning",
			Summary: []ResponsesReasoningSummary{
				{
					Type: "summary_text",
					Text: chatResponse.Message.Thinking,
				},
			},
		})
	}

	if len(chatResponse.Message.ToolCalls) > 0 {
		toolCalls := ToToolCalls(chatResponse.Message.ToolCalls)
		for i, tc := range toolCalls {
			output = append(output, ResponsesOutputItem{
				ID:        fmt.Sprintf("fc_%s_%d", responseID, i),
				Type:      "function_call",
				Status:    "completed",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	} else {
		output = append(output, ResponsesOutputItem{
			ID:     itemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []ResponsesOutputContent{
				{
					Type:        "output_text",
					Text:        chatResponse.Message.Content,
					Annotations: []any{},
					Logprobs:    []any{},
				},
			},
		})
	}

	var instructions *string
	if request.Instructions != "" {
		instructions = &request.Instructions
	}

	// Build truncation with default
	truncation := "disabled"
	if request.Truncation != nil {
		truncation = *request.Truncation
	}

	tools := request.Tools
	if tools == nil {
		tools = []ResponsesTool{}
	}

	text := ResponsesTextField{
		Format: ResponsesTextFormat{Type: "text"},
	}
	if request.Text != nil && request.Text.Format != nil {
		text.Format = *request.Text.Format
	}

	// Build reasoning output from request.
	// If the local model produced thinking output but the request did not
	// specify a reasoning.summary preference, default to "concise" so that
	// Codex CLI surfaces the reasoning block rather than hiding it.
	var reasoning *ResponsesReasoningOutput
	effectiveSummary := request.Reasoning.Summary
	if chatResponse.Message.Thinking != "" && effectiveSummary == "" {
		effectiveSummary = "concise"
	}
	if request.Reasoning.Effort != "" || effectiveSummary != "" {
		reasoning = &ResponsesReasoningOutput{}
		if request.Reasoning.Effort != "" {
			reasoning.Effort = &request.Reasoning.Effort
		}
		if effectiveSummary != "" {
			reasoning.Summary = &effectiveSummary
		}
	}

	return ResponsesResponse{
		ID:                 responseID,
		Object:             "response",
		CreatedAt:          chatResponse.CreatedAt.Unix(),
		CompletedAt:        nil, // Set by middleware when writing final response
		Status:             "completed",
		IncompleteDetails:  nil, // Only populated if response incomplete
		Model:              model,
		PreviousResponseID: nil, // Not supported
		Instructions:       instructions,
		Output:             output,
		Error:              nil, // Only populated on failure
		Tools:              tools,
		ToolChoice: func() any {
			if len(request.ToolChoice) > 0 {
				var v any
				if json.Unmarshal(request.ToolChoice, &v) == nil {
					return v
				}
			}
			return "auto"
		}(), // Echo back the request's tool_choice, defaulting to "auto"
		Truncation:        truncation,
		ParallelToolCalls: true, // Default value
		Text:              text,
		TopP:              derefFloat64(request.TopP, 1.0),
		PresencePenalty:   0, // Default value
		FrequencyPenalty:  0, // Default value
		TopLogprobs:       0, // Default value
		Temperature:       derefFloat64(request.Temperature, 1.0),
		Reasoning:         reasoning,
		Usage: &ResponsesUsage{
			InputTokens:        chatResponse.PromptEvalCount,
			OutputTokens:       chatResponse.EvalCount,
			TotalTokens:        chatResponse.PromptEvalCount + chatResponse.EvalCount,
			InputTokensDetails: ResponsesInputTokensDetails{CachedTokens: 0},
			// Ollama does not report thinking tokens separately; estimate from
			// the thinking text length using the standard ~4 chars/token heuristic.
			OutputTokensDetails: ResponsesOutputTokensDetails{
				ReasoningTokens: len([]rune(chatResponse.Message.Thinking)) / 4,
			},
		},
		MaxOutputTokens:  request.MaxOutputTokens,
		MaxToolCalls:     nil,   // Not supported
		Store:            false, // We don't store responses
		Background:       request.Background,
		ServiceTier:      "default", // Default value
		Metadata:         map[string]any{},
		SafetyIdentifier: nil, // Not supported
		PromptCacheKey:   nil, // Not supported
	}
}

// Streaming events: <https://platform.openai.com/docs/api-reference/responses-streaming>

// ResponsesStreamEvent represents a single Server-Sent Event for the Responses API.
type ResponsesStreamEvent struct {
	Event string // The event type (e.g., "response.created")
	Data  any    // The event payload (will be JSON-marshaled)
}

// ResponsesStreamConverter converts api.ChatResponse objects to Responses API
// streaming events. It maintains state across multiple calls to handle the
// streaming event sequence correctly.
type ResponsesStreamConverter struct {
	// Configuration (immutable after creation)
	responseID string
	itemID     string
	model      string
	request    ResponsesRequest

	// State tracking (mutated across Process calls)
	firstWrite      bool
	outputIndex     int
	contentIndex    int
	contentStarted  bool
	toolCallsSent   bool
	accumulatedText string
	sequenceNumber  int

	// Reasoning/thinking state
	accumulatedThinking string
	reasoningItemID     string
	reasoningStarted    bool
	reasoningDone       bool

	// Tool calls state (for final output)
	toolCallItems []map[string]any
}

// newEvent creates a ResponsesStreamEvent with the sequence number included in the data.
func (c *ResponsesStreamConverter) newEvent(eventType string, data map[string]any) ResponsesStreamEvent {
	data["type"] = eventType
	data["sequence_number"] = c.sequenceNumber
	c.sequenceNumber++
	return ResponsesStreamEvent{
		Event: eventType,
		Data:  data,
	}
}

// NewResponsesStreamConverter creates a new converter with the given configuration.
func NewResponsesStreamConverter(responseID, itemID, model string, request ResponsesRequest) *ResponsesStreamConverter {
	return &ResponsesStreamConverter{
		responseID: responseID,
		itemID:     itemID,
		model:      model,
		request:    request,
		firstWrite: true,
	}
}

// ResetFirstWrite re-enables the initial response.created / response.in_progress
// events so they are emitted again after the web search agent loop takes over
// the write path. This is needed because the converter may have already been
// primed by an earlier (discarded) write.
func (c *ResponsesStreamConverter) ResetFirstWrite() {
	c.firstWrite = true
}

// Process takes a ChatResponse and returns the events that should be emitted.
// Events are returned in order. The caller is responsible for serializing
// and sending these events.
func (c *ResponsesStreamConverter) Process(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	hasToolCalls := len(r.Message.ToolCalls) > 0
	hasThinking := r.Message.Thinking != ""

	// First chunk - emit initial events
	if c.firstWrite {
		c.firstWrite = false
		events = append(events, c.createResponseCreatedEvent())
		events = append(events, c.createResponseInProgressEvent())
	}

	// Handle reasoning/thinking (before other content)
	if hasThinking {
		events = append(events, c.processThinking(r.Message.Thinking)...)
	}

	// Handle tool calls
	if hasToolCalls {
		events = append(events, c.processToolCalls(r.Message.ToolCalls)...)
		c.toolCallsSent = true
	}

	// Handle text content (only if no tool calls)
	if !hasToolCalls && !c.toolCallsSent && r.Message.Content != "" {
		events = append(events, c.processTextContent(r.Message.Content)...)
	}

	// Done - emit closing events
	if r.Done {
		events = append(events, c.processCompletion(r)...)
	}

	return events
}

// buildResponseObject creates a full response object with all required fields for streaming events.
func (c *ResponsesStreamConverter) buildResponseObject(status string, output []any, usage map[string]any) map[string]any {
	var instructions any = nil
	if c.request.Instructions != "" {
		instructions = c.request.Instructions
	}

	truncation := "disabled"
	if c.request.Truncation != nil {
		truncation = *c.request.Truncation
	}

	var tools []any
	if c.request.Tools != nil {
		for _, t := range c.request.Tools {
			tools = append(tools, map[string]any{
				"type":        t.Type,
				"name":        t.Name,
				"description": t.Description,
				"strict":      t.Strict,
				"parameters":  t.Parameters,
			})
		}
	}
	if tools == nil {
		tools = []any{}
	}

	textFormat := map[string]any{"type": "text"}
	if c.request.Text != nil && c.request.Text.Format != nil {
		textFormat = map[string]any{
			"type": c.request.Text.Format.Type,
		}
		if c.request.Text.Format.Name != "" {
			textFormat["name"] = c.request.Text.Format.Name
		}
		if c.request.Text.Format.Schema != nil {
			textFormat["schema"] = c.request.Text.Format.Schema
		}
		if c.request.Text.Format.Strict != nil {
			textFormat["strict"] = *c.request.Text.Format.Strict
		}
	}

	var reasoning any = nil
	if c.request.Reasoning.Effort != "" || c.request.Reasoning.Summary != "" {
		r := map[string]any{}
		if c.request.Reasoning.Effort != "" {
			r["effort"] = c.request.Reasoning.Effort
		} else {
			r["effort"] = nil
		}
		if c.request.Reasoning.Summary != "" {
			r["summary"] = c.request.Reasoning.Summary
		} else {
			r["summary"] = nil
		}
		reasoning = r
	}

	// Build top_p and temperature with defaults
	topP := 1.0
	if c.request.TopP != nil {
		topP = *c.request.TopP
	}
	temperature := 1.0
	if c.request.Temperature != nil {
		temperature = *c.request.Temperature
	}

	return map[string]any{
		"id":                   c.responseID,
		"object":               "response",
		"created_at":           time.Now().Unix(),
		"completed_at":         nil,
		"status":               status,
		"incomplete_details":   nil,
		"model":                c.model,
		"previous_response_id": nil,
		"instructions":         instructions,
		"output":               output,
		"error":                nil,
		"tools":                tools,
		"tool_choice": func() any {
			if len(c.request.ToolChoice) > 0 {
				var v any
				if json.Unmarshal(c.request.ToolChoice, &v) == nil {
					return v
				}
			}
			return "auto"
		}(),
		"truncation":          truncation,
		"parallel_tool_calls": true,
		"text":                map[string]any{"format": textFormat},
		"top_p":               topP,
		"presence_penalty":    0,
		"frequency_penalty":   0,
		"top_logprobs":        0,
		"temperature":         temperature,
		"reasoning":           reasoning,
		"usage":               usage,
		"max_output_tokens":   c.request.MaxOutputTokens,
		"max_tool_calls":      nil,
		"store":               false,
		"background":          c.request.Background,
		"service_tier":        "default",
		"metadata":            map[string]any{},
		"safety_identifier":   nil,
		"prompt_cache_key":    nil,
	}
}

func (c *ResponsesStreamConverter) createResponseCreatedEvent() ResponsesStreamEvent {
	return c.newEvent("response.created", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) createResponseInProgressEvent() ResponsesStreamEvent {
	return c.newEvent("response.in_progress", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) processThinking(thinking string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Start reasoning item if not started
	if !c.reasoningStarted {
		c.reasoningStarted = true
		c.reasoningItemID = fmt.Sprintf("rs_%d", rand.Intn(999999))

		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.reasoningItemID,
				"type":    "reasoning",
				"summary": []any{},
			},
		}))
	}

	// Accumulate thinking
	c.accumulatedThinking += thinking

	// Emit delta
	events = append(events, c.newEvent("response.reasoning_summary_text.delta", map[string]any{
		"item_id":       c.reasoningItemID,
		"output_index":  c.outputIndex,
		"summary_index": 0,
		"delta":         thinking,
	}))

	// TODO(drifkin): consider adding
	// [`response.reasoning_text.delta`](https://platform.openai.com/docs/api-reference/responses-streaming/response/reasoning_text/delta),
	// but need to do additional research to understand how it's used and how
	// widely supported it is

	return events
}

func (c *ResponsesStreamConverter) finishReasoning() []ResponsesStreamEvent {
	if !c.reasoningStarted || c.reasoningDone {
		return nil
	}
	c.reasoningDone = true

	events := []ResponsesStreamEvent{
		c.newEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id":       c.reasoningItemID,
			"output_index":  c.outputIndex,
			"summary_index": 0,
			"text":          c.accumulatedThinking,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.reasoningItemID,
				"type":    "reasoning",
				"summary": []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
			},
		}),
	}

	c.outputIndex++
	return events
}

func (c *ResponsesStreamConverter) processToolCalls(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	converted := ToToolCalls(toolCalls)

	for i, tc := range converted {
		fcItemID := fmt.Sprintf("fc_%d_%d", rand.Intn(999999), i)

		// Store for final output (with status: completed)
		toolCallItem := map[string]any{
			"id":        fcItemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		}
		c.toolCallItems = append(c.toolCallItems, toolCallItem)

		// response.output_item.added for function call
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": "",
			},
		}))

		// response.function_call_arguments.delta
		if tc.Function.Arguments != "" {
			events = append(events, c.newEvent("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcItemID,
				"output_index": c.outputIndex + i,
				"delta":        tc.Function.Arguments,
			}))
		}

		// response.function_call_arguments.done
		events = append(events, c.newEvent("response.function_call_arguments.done", map[string]any{
			"item_id":      fcItemID,
			"output_index": c.outputIndex + i,
			"arguments":    tc.Function.Arguments,
		}))

		// response.output_item.done for function call
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
			},
		}))
	}

	return events
}

// WebSearchCallEvents emits the Responses API streaming events for a
// web_search_call item: output_item.added (in_progress), output_item.done
// (completed), and the final item stored for the response.completed payload.
// The caller (ResponsesWebSearchWriter) must call this before the final
// text-content events so that output indices are correct.
func (c *ResponsesStreamConverter) WebSearchCallEvents(wsItemID, query string, results []map[string]any) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// response.output_item.added (status: in_progress)
	events = append(events, c.newEvent("response.output_item.added", map[string]any{
		"output_index": c.outputIndex,
		"item": map[string]any{
			"id":     wsItemID,
			"type":   "web_search_call",
			"status": "in_progress",
			"action": map[string]any{
				"type":  "search",
				"query": query,
			},
		},
	}))

	// response.output_item.done (status: completed)
	wsItem := map[string]any{
		"id":     wsItemID,
		"type":   "web_search_call",
		"status": "completed",
		"action": map[string]any{
			"type":  "search",
			"query": query,
		},
	}
	if len(results) > 0 {
		wsItem["results"] = results
	}
	events = append(events, c.newEvent("response.output_item.done", map[string]any{
		"output_index": c.outputIndex,
		"item":         wsItem,
	}))

	// Store for buildFinalOutput / response.completed
	c.toolCallItems = append(c.toolCallItems, wsItem)
	c.outputIndex++
	return events
}

// FileSearchCallEvents emits the Responses API streaming events for a
// file_search_call item: output_item.added (in_progress), output_item.done
// (completed with results). The caller (ResponsesFileSearchWriter) must call
// this before the final text-content events so that output indices are correct.
//
// chunks is []middleware.FileChunkResult but typed as []any here to avoid an
// import cycle; the caller passes the slice as []any.
func (c *ResponsesStreamConverter) FileSearchCallEvents(fsItemID, query string, chunks any) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent
	// response.output_item.added (status: in_progress)
	events = append(events, c.newEvent("response.output_item.added", map[string]any{
		"output_index": c.outputIndex,
		"item": map[string]any{
			"id":     fsItemID,
			"type":   "file_search_call",
			"status": "in_progress",
			"query":  query,
		},
	}))
	// response.output_item.done (status: completed)
	fsItem := map[string]any{
		"id":     fsItemID,
		"type":   "file_search_call",
		"status": "completed",
		"query":  query,
		"results": chunks,
	}
	events = append(events, c.newEvent("response.output_item.done", map[string]any{
		"output_index": c.outputIndex,
		"item":         fsItem,
	}))
	// Store for buildFinalOutput / response.completed
	c.toolCallItems = append(c.toolCallItems, fsItem)
	c.outputIndex++
	return events
}

func (c *ResponsesStreamConverter) processTextContent(content string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	// Emit output item and content part for first text content
	if !c.contentStarted {
		c.contentStarted = true

		// response.output_item.added
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.itemID,
				"type":    "message",
				"status":  "in_progress",
				"role":    "assistant",
				"phase":   "final_answer",
				"content": []any{},
			},
		}))

		// response.content_part.added
		events = append(events, c.newEvent("response.content_part.added", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": c.contentIndex,
			"part": map[string]any{
				"type":        "output_text",
				"text":        "",
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}))
	}

	// Accumulate text
	c.accumulatedText += content

	// Emit content delta
	events = append(events, c.newEvent("response.output_text.delta", map[string]any{
		"item_id":       c.itemID,
		"output_index":  c.outputIndex,
		"content_index": 0,
		"delta":         content,
		"logprobs":      []any{},
	}))

	return events
}

func (c *ResponsesStreamConverter) buildFinalOutput() []any {
	var output []any

	// Add reasoning item if present
	if c.reasoningStarted {
		output = append(output, map[string]any{
			"id":      c.reasoningItemID,
			"type":    "reasoning",
			"summary": []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
		})
	}

	// Add tool calls if present
	if len(c.toolCallItems) > 0 {
		for _, item := range c.toolCallItems {
			output = append(output, item)
		}
	} else if c.contentStarted {
		// Add message item if we had text content
		output = append(output, map[string]any{
			"id":     c.itemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"phase":  "final_answer",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			}},
		})
	}

	return output
}

func (c *ResponsesStreamConverter) processCompletion(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning if not done
	events = append(events, c.finishReasoning()...)

	// Emit text completion events if we had text content
	if !c.toolCallsSent && c.contentStarted {
		// response.output_text.done
		events = append(events, c.newEvent("response.output_text.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"text":          c.accumulatedText,
			"logprobs":      []any{},
		}))

		// response.content_part.done
		events = append(events, c.newEvent("response.content_part.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}))

		// response.output_item.done
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":     c.itemID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
				"phase":  "final_answer",
				"content": []map[string]any{{
					"type":        "output_text",
					"text":        c.accumulatedText,
					"annotations": []any{},
					"logprobs":    []any{},
				}},
			},
		}))
	}

	// response.completed
	usage := map[string]any{
		"input_tokens":  r.PromptEvalCount,
		"output_tokens": r.EvalCount,
		"total_tokens":  r.PromptEvalCount + r.EvalCount,
		"input_tokens_details": map[string]any{
			"cached_tokens": 0,
		},
		"output_tokens_details": map[string]any{
			// Estimate reasoning tokens from accumulated thinking text.
			"reasoning_tokens": len([]rune(c.accumulatedThinking)) / 4,
		},
	}
	response := c.buildResponseObject("completed", c.buildFinalOutput(), usage)
	response["completed_at"] = time.Now().Unix()
	events = append(events, c.newEvent("response.completed", map[string]any{
		"response": response,
	}))

	return events
}
