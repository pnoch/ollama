package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// ToolChoiceKind represents the parsed kind of a tool_choice value.
type ToolChoiceKind int

const (
	ToolChoiceAuto     ToolChoiceKind = iota // "auto" or unset
	ToolChoiceNone                           // "none"
	ToolChoiceRequired                       // "required"
	ToolChoiceNamed                          // {"type":"function","name":"X"} or {"type":"function","function":{"name":"X"}}
)

// ParsedToolChoice holds the result of parsing a raw tool_choice JSON value.
type ParsedToolChoice struct {
	Kind      ToolChoiceKind
	FuncName  string // only set when Kind == ToolChoiceNamed
}

// ParseToolChoice parses a raw JSON tool_choice value into a ParsedToolChoice.
// A nil or empty raw value is treated as ToolChoiceAuto.
func ParseToolChoice(raw json.RawMessage) ParsedToolChoice {
	if len(raw) == 0 {
		return ParsedToolChoice{Kind: ToolChoiceAuto}
	}
	// Try string form first: "auto", "none", "required"
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "none":
			return ParsedToolChoice{Kind: ToolChoiceNone}
		case "required":
			return ParsedToolChoice{Kind: ToolChoiceRequired}
		default:
			return ParsedToolChoice{Kind: ToolChoiceAuto}
		}
	}
	// Try both object forms:
	//   Responses API (flat):  {"type":"function","name":"X"}
	//   Chat API (nested):     {"type":"function","function":{"name":"X"}}
	// The nested form takes precedence when both name fields are present.
	var obj struct {
		Type     string `json:"type"`
		Name     string `json:"name"`     // Responses API flat form
		Function struct {
			Name string `json:"name"`
		} `json:"function"` // Chat API nested form
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Type == "function" {
		if obj.Function.Name != "" {
			// Chat API nested form: {"type":"function","function":{"name":"X"}}
			return ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: obj.Function.Name}
		}
		if obj.Name != "" {
			// Responses API flat form: {"type":"function","name":"X"}
			return ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: obj.Name}
		}
	}
	return ParsedToolChoice{Kind: ToolChoiceAuto}
}

// ToolChoiceGrammarSchema builds a JSON Schema that constrains the model's
// output to a valid tool call for the given tool_choice.
//
//   - ToolChoiceRequired: the model must call any one of the provided tools.
//   - ToolChoiceNamed:    the model must call the specific named tool.
//
// The returned schema is suitable for passing to llama.SchemaToGrammar.
// Returns nil if no grammar is needed (auto/none) or if no matching tools exist.
func ToolChoiceGrammarSchema(choice ParsedToolChoice, tools []api.Tool) []byte {
	switch choice.Kind {
	case ToolChoiceRequired:
		return toolUnionSchema(tools)
	case ToolChoiceNamed:
		for _, t := range tools {
			if t.Function.Name == choice.FuncName {
				return toolCallSchema(t)
			}
		}
		return nil
	default:
		return nil
	}
}

// toolCallSchema builds a JSON Schema for a single tool call object:
//
//	{"name": "<tool>", "arguments": <tool-params-schema>}
func toolCallSchema(t api.Tool) []byte {
	paramsSchema := toolParamsSchema(t.Function.Parameters)
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":  "string",
				"const": t.Function.Name,
			},
			"arguments": paramsSchema,
		},
		"required":             []string{"name", "arguments"},
		"additionalProperties": false,
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	return b
}

// toolUnionSchema builds a JSON Schema that accepts any one of the provided
// tool call objects using "anyOf".
func toolUnionSchema(tools []api.Tool) []byte {
	if len(tools) == 0 {
		return nil
	}
	if len(tools) == 1 {
		return toolCallSchema(tools[0])
	}
	anyOf := make([]json.RawMessage, 0, len(tools))
	for _, t := range tools {
		s := toolCallSchema(t)
		if s != nil {
			anyOf = append(anyOf, json.RawMessage(s))
		}
	}
	if len(anyOf) == 0 {
		return nil
	}
	schema := map[string]any{
		"anyOf": anyOf,
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	return b
}

// toolParamsSchema converts a ToolFunctionParameters into a JSON Schema map.
func toolParamsSchema(params api.ToolFunctionParameters) map[string]any {
	schema := map[string]any{
		"type": "object",
	}
	if params.Properties != nil && params.Properties.Len() > 0 {
		props := map[string]any{}
		for name, prop := range params.Properties.All() {
			props[name] = toolPropertySchema(prop)
		}
		schema["properties"] = props
	}
	if len(params.Required) > 0 {
		schema["required"] = params.Required
	}
	schema["additionalProperties"] = false
	return schema
}

// toolPropertySchema converts a ToolProperty into a JSON Schema map.
func toolPropertySchema(prop api.ToolProperty) map[string]any {
	schema := map[string]any{}
	if len(prop.AnyOf) > 0 {
		anyOf := make([]map[string]any, 0, len(prop.AnyOf))
		for _, p := range prop.AnyOf {
			anyOf = append(anyOf, toolPropertySchema(p))
		}
		schema["anyOf"] = anyOf
		return schema
	}
	if len(prop.Type) > 0 {
		if len(prop.Type) == 1 {
			schema["type"] = prop.Type[0]
		} else {
			schema["type"] = []string(prop.Type)
		}
	}
	if prop.Description != "" {
		schema["description"] = prop.Description
	}
	if len(prop.Enum) > 0 {
		schema["enum"] = prop.Enum
	}
	if prop.Properties != nil && prop.Properties.Len() > 0 {
		props := map[string]any{}
		for name, p := range prop.Properties.All() {
			props[name] = toolPropertySchema(p)
		}
		schema["properties"] = props
	}
	return schema
}

// ToolChoiceSystemSuffix returns a system prompt suffix to reinforce tool_choice
// for models that may not fully respect grammar constraints alone.
// Returns an empty string when no suffix is needed.
func ToolChoiceSystemSuffix(choice ParsedToolChoice) string {
	switch choice.Kind {
	case ToolChoiceRequired:
		return "You must respond by calling one of the provided tools. Do not reply with plain text."
	case ToolChoiceNamed:
		return fmt.Sprintf(
			"You must respond by calling the %q tool. Do not reply with plain text or call any other tool.",
			choice.FuncName,
		)
	default:
		return ""
	}
}

// ToolChoiceFilterTools returns the subset of tools that should be presented
// to the model given the tool_choice. For "none" it returns nil (no tools).
// For "named" it returns only the named tool. For "auto"/"required" it returns
// all tools unchanged.
func ToolChoiceFilterTools(choice ParsedToolChoice, allTools []api.Tool) []api.Tool {
	switch choice.Kind {
	case ToolChoiceNone:
		return nil
	case ToolChoiceNamed:
		for _, t := range allTools {
			if t.Function.Name == choice.FuncName {
				return []api.Tool{t}
			}
		}
		return nil
	default:
		return allTools
	}
}

// ToolChoiceDescription returns a human-readable description of the tool_choice
// for logging purposes.
func ToolChoiceDescription(choice ParsedToolChoice) string {
	switch choice.Kind {
	case ToolChoiceNone:
		return "none"
	case ToolChoiceRequired:
		return "required"
	case ToolChoiceNamed:
		return fmt.Sprintf("function:%s", choice.FuncName)
	default:
		return "auto"
	}
}

// injectToolChoiceSystemMessage appends a tool_choice enforcement message to
// the message list when needed. The message is inserted as the last system
// message before any user/assistant turns.
func InjectToolChoiceSystemMessage(msgs []api.Message, choice ParsedToolChoice) []api.Message {
	suffix := ToolChoiceSystemSuffix(choice)
	if suffix == "" {
		return msgs
	}
	// Find the last system message and append to it, or prepend a new one
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "system" || msgs[i].Role == "developer" {
			msgs[i].Content = strings.TrimRight(msgs[i].Content, " \n") + "\n\n" + suffix
			return msgs
		}
	}
	// No system message found — prepend one
	return append([]api.Message{{Role: "system", Content: suffix}}, msgs...)
}
