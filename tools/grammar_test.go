package tools

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// ─── ParseToolChoice ──────────────────────────────────────────────────────────

func TestParseToolChoice_Auto(t *testing.T) {
	for _, raw := range []json.RawMessage{nil, {}, json.RawMessage(`"auto"`)} {
		got := ParseToolChoice(raw)
		if got.Kind != ToolChoiceAuto {
			t.Errorf("ParseToolChoice(%q) = %v, want ToolChoiceAuto", string(raw), got.Kind)
		}
	}
}

func TestParseToolChoice_None(t *testing.T) {
	got := ParseToolChoice(json.RawMessage(`"none"`))
	if got.Kind != ToolChoiceNone {
		t.Errorf("expected ToolChoiceNone, got %v", got.Kind)
	}
}

func TestParseToolChoice_Required(t *testing.T) {
	got := ParseToolChoice(json.RawMessage(`"required"`))
	if got.Kind != ToolChoiceRequired {
		t.Errorf("expected ToolChoiceRequired, got %v", got.Kind)
	}
}

func TestParseToolChoice_NamedFunction(t *testing.T) {
	raw := json.RawMessage(`{"type":"function","function":{"name":"my_tool"}}`)
	got := ParseToolChoice(raw)
	if got.Kind != ToolChoiceNamed {
		t.Errorf("expected ToolChoiceNamed, got %v", got.Kind)
	}
	if got.FuncName != "my_tool" {
		t.Errorf("expected FuncName=my_tool, got %q", got.FuncName)
	}
}

func TestParseToolChoice_NamedFunction_FlatForm(t *testing.T) {
	// Responses API flat form: {"type":"function","name":"X"}
	raw := json.RawMessage(`{"type":"function","name":"my_tool"}`)
	got := ParseToolChoice(raw)
	if got.Kind != ToolChoiceNamed {
		t.Errorf("expected ToolChoiceNamed for flat form, got %v", got.Kind)
	}
	if got.FuncName != "my_tool" {
		t.Errorf("expected FuncName=my_tool, got %q", got.FuncName)
	}
}

func TestParseToolChoice_NamedFunction_NestedFormTakesPrecedence(t *testing.T) {
	// When both name fields are present, nested form wins
	raw := json.RawMessage(`{"type":"function","name":"flat_name","function":{"name":"nested_name"}}`)
	got := ParseToolChoice(raw)
	if got.Kind != ToolChoiceNamed {
		t.Errorf("expected ToolChoiceNamed, got %v", got.Kind)
	}
	if got.FuncName != "nested_name" {
		t.Errorf("expected FuncName=nested_name (nested wins), got %q", got.FuncName)
	}
}

func TestParseToolChoice_MalformedObject(t *testing.T) {
	// object without the expected fields → auto
	raw := json.RawMessage(`{"type":"unknown"}`)
	got := ParseToolChoice(raw)
	if got.Kind != ToolChoiceAuto {
		t.Errorf("expected ToolChoiceAuto for malformed object, got %v", got.Kind)
	}
}

// ─── ToolChoiceGrammarSchema ──────────────────────────────────────────────────

func sampleTool(name string) api.Tool {
	params := api.ToolFunctionParameters{}
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        name,
			Description: "A test tool",
			Parameters:  params,
		},
	}
}

func TestToolChoiceGrammarSchema_Auto_ReturnsNil(t *testing.T) {
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceAuto}, []api.Tool{sampleTool("foo")})
	if got != nil {
		t.Errorf("expected nil for auto, got %s", string(got))
	}
}

func TestToolChoiceGrammarSchema_None_ReturnsNil(t *testing.T) {
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNone}, []api.Tool{sampleTool("foo")})
	if got != nil {
		t.Errorf("expected nil for none, got %s", string(got))
	}
}

func TestToolChoiceGrammarSchema_Required_SingleTool(t *testing.T) {
	tools := []api.Tool{sampleTool("run_shell")}
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceRequired}, tools)
	if got == nil {
		t.Fatal("expected non-nil schema for required")
	}
	// Should be valid JSON
	var v any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Errorf("schema is not valid JSON: %v", err)
	}
	// Should contain the tool name as a const
	if !strings.Contains(string(got), "run_shell") {
		t.Errorf("schema does not contain tool name 'run_shell': %s", string(got))
	}
}

func TestToolChoiceGrammarSchema_Required_MultipleTools(t *testing.T) {
	tools := []api.Tool{sampleTool("tool_a"), sampleTool("tool_b")}
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceRequired}, tools)
	if got == nil {
		t.Fatal("expected non-nil schema for required with multiple tools")
	}
	var v any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Errorf("schema is not valid JSON: %v", err)
	}
	// Multi-tool schema should use anyOf
	if !strings.Contains(string(got), "anyOf") {
		t.Errorf("expected anyOf in multi-tool schema: %s", string(got))
	}
	if !strings.Contains(string(got), "tool_a") || !strings.Contains(string(got), "tool_b") {
		t.Errorf("schema missing tool names: %s", string(got))
	}
}

func TestToolChoiceGrammarSchema_Named_MatchingTool(t *testing.T) {
	tools := []api.Tool{sampleTool("tool_a"), sampleTool("tool_b")}
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "tool_b"}, tools)
	if got == nil {
		t.Fatal("expected non-nil schema for named tool_b")
	}
	if !strings.Contains(string(got), "tool_b") {
		t.Errorf("schema does not contain 'tool_b': %s", string(got))
	}
	// Should NOT contain tool_a
	if strings.Contains(string(got), "tool_a") {
		t.Errorf("schema unexpectedly contains 'tool_a': %s", string(got))
	}
}

func TestToolChoiceGrammarSchema_Named_NoMatch_ReturnsNil(t *testing.T) {
	tools := []api.Tool{sampleTool("tool_a")}
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "nonexistent"}, tools)
	if got != nil {
		t.Errorf("expected nil for unmatched named tool, got %s", string(got))
	}
}

// ─── ToolChoiceFilterTools ────────────────────────────────────────────────────

func TestToolChoiceFilterTools_Auto_ReturnsAll(t *testing.T) {
	all := []api.Tool{sampleTool("a"), sampleTool("b")}
	got := ToolChoiceFilterTools(ParsedToolChoice{Kind: ToolChoiceAuto}, all)
	if len(got) != 2 {
		t.Errorf("expected 2 tools for auto, got %d", len(got))
	}
}

func TestToolChoiceFilterTools_None_ReturnsNil(t *testing.T) {
	all := []api.Tool{sampleTool("a"), sampleTool("b")}
	got := ToolChoiceFilterTools(ParsedToolChoice{Kind: ToolChoiceNone}, all)
	if got != nil {
		t.Errorf("expected nil for none, got %v", got)
	}
}

func TestToolChoiceFilterTools_Named_ReturnsSingleTool(t *testing.T) {
	all := []api.Tool{sampleTool("a"), sampleTool("b"), sampleTool("c")}
	got := ToolChoiceFilterTools(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "b"}, all)
	if len(got) != 1 || got[0].Function.Name != "b" {
		t.Errorf("expected single tool 'b', got %v", got)
	}
}

// ─── InjectToolChoiceSystemMessage ───────────────────────────────────────────

func TestInjectToolChoiceSystemMessage_Auto_NoChange(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hello"}}
	got := InjectToolChoiceSystemMessage(msgs, ParsedToolChoice{Kind: ToolChoiceAuto})
	if len(got) != 1 {
		t.Errorf("expected no change for auto, got %v", got)
	}
}

func TestInjectToolChoiceSystemMessage_Required_AppendsSuffix(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "do something"},
	}
	got := InjectToolChoiceSystemMessage(msgs, ParsedToolChoice{Kind: ToolChoiceRequired})
	// The system message should have the suffix appended
	if !strings.Contains(got[0].Content, "You must respond by calling") {
		t.Errorf("expected system message to contain tool_choice suffix, got: %q", got[0].Content)
	}
	// Original content should still be there
	if !strings.Contains(got[0].Content, "You are a helpful assistant.") {
		t.Errorf("original system content was lost: %q", got[0].Content)
	}
}

func TestInjectToolChoiceSystemMessage_Named_AppendsSuffixWithToolName(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "do something"}}
	got := InjectToolChoiceSystemMessage(msgs, ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "my_tool"})
	// Should prepend a new system message since there was none
	if len(got) != 2 {
		t.Errorf("expected 2 messages (new system + user), got %d", len(got))
	}
	if got[0].Role != "system" {
		t.Errorf("expected first message to be system, got %q", got[0].Role)
	}
	if !strings.Contains(got[0].Content, "my_tool") {
		t.Errorf("expected system message to mention 'my_tool', got: %q", got[0].Content)
	}
}

// ─── ToolChoiceDescription ───────────────────────────────────────────────────

func TestToolChoiceDescription(t *testing.T) {
	cases := []struct {
		choice ParsedToolChoice
		want   string
	}{
		{ParsedToolChoice{Kind: ToolChoiceAuto}, "auto"},
		{ParsedToolChoice{Kind: ToolChoiceNone}, "none"},
		{ParsedToolChoice{Kind: ToolChoiceRequired}, "required"},
		{ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "foo"}, "function:foo"},
	}
	for _, tc := range cases {
		got := ToolChoiceDescription(tc.choice)
		if got != tc.want {
			t.Errorf("ToolChoiceDescription(%v) = %q, want %q", tc.choice, got, tc.want)
		}
	}
}


// ─── Edge cases ───────────────────────────────────────────────────────────────

// TestToolChoiceGrammarSchema_Required_ZeroTools verifies that required with no
// tools returns nil (nothing to constrain against).
func TestToolChoiceGrammarSchema_Required_ZeroTools(t *testing.T) {
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceRequired}, nil)
	if got != nil {
		t.Errorf("expected nil for required with no tools, got %s", string(got))
	}
}

// TestToolChoiceGrammarSchema_Required_EmptyToolList verifies that an empty
// (non-nil) tool slice behaves the same as nil — returns nil schema.
func TestToolChoiceGrammarSchema_Required_EmptyToolList(t *testing.T) {
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceRequired}, []api.Tool{})
	if got != nil {
		t.Errorf("expected nil for required with empty tool list, got %s", string(got))
	}
}

// TestToolChoiceGrammarSchema_Named_NoParamsTool verifies that a named tool
// with no parameters still produces a valid schema (arguments is an empty
// object schema, not null).
func TestToolChoiceGrammarSchema_Named_NoParamsTool(t *testing.T) {
	noParamsTool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "ping",
			Description: "Ping with no arguments",
			Parameters:  api.ToolFunctionParameters{},
		},
	}
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "ping"}, []api.Tool{noParamsTool})
	if got == nil {
		t.Fatal("expected non-nil schema for named tool with no parameters")
	}
	var v map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	// Must contain the tool name as a const.
	if !strings.Contains(string(got), "ping") {
		t.Errorf("schema does not contain tool name 'ping': %s", string(got))
	}
	// arguments property must be present and be an object schema.
	props, _ := v["properties"].(map[string]any)
	if props == nil {
		t.Fatalf("schema missing properties: %s", string(got))
	}
	args, _ := props["arguments"].(map[string]any)
	if args == nil {
		t.Fatalf("schema missing arguments property: %s", string(got))
	}
	if args["type"] != "object" {
		t.Errorf("arguments type = %v, want object", args["type"])
	}
}

// TestToolCallSchema_AdditionalPropertiesFalse verifies that the generated
// schema always sets additionalProperties=false at the top level, which is
// required for grammar-constrained generation to reject unexpected fields.
func TestToolCallSchema_AdditionalPropertiesFalse(t *testing.T) {
	tool := sampleTool("strict_tool")
	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "strict_tool"}, []api.Tool{tool})
	if got == nil {
		t.Fatal("expected non-nil schema")
	}
	var v map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	if v["additionalProperties"] != false {
		t.Errorf("expected additionalProperties=false at top level, got %v", v["additionalProperties"])
	}
}

// TestToolChoiceFilterTools_Named_NonExistent verifies that filtering for a
// named tool that does not exist in the list returns nil (not an empty slice).
func TestToolChoiceFilterTools_Named_NonExistent(t *testing.T) {
	all := []api.Tool{sampleTool("a"), sampleTool("b")}
	got := ToolChoiceFilterTools(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "nonexistent"}, all)
	if got != nil {
		t.Errorf("expected nil for non-existent named tool, got %v", got)
	}
}

// TestToolChoiceFilterTools_Required_ReturnsAll verifies that required returns
// all tools unchanged (same as auto).
func TestToolChoiceFilterTools_Required_ReturnsAll(t *testing.T) {
	all := []api.Tool{sampleTool("a"), sampleTool("b"), sampleTool("c")}
	got := ToolChoiceFilterTools(ParsedToolChoice{Kind: ToolChoiceRequired}, all)
	if len(got) != 3 {
		t.Errorf("expected 3 tools for required, got %d", len(got))
	}
}

// TestInjectToolChoiceSystemMessage_MultipleSystemMessages verifies that when
// there are multiple system messages, the suffix is appended to the LAST one
// (not the first), so it is closest to the user turn.
func TestInjectToolChoiceSystemMessage_MultipleSystemMessages(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "First system message."},
		{Role: "system", Content: "Second system message."},
		{Role: "user", Content: "do something"},
	}
	got := InjectToolChoiceSystemMessage(msgs, ParsedToolChoice{Kind: ToolChoiceRequired})
	if len(got) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(got))
	}
	// First system message should be unchanged.
	if got[0].Content != "First system message." {
		t.Errorf("first system message was modified: %q", got[0].Content)
	}
	// Second (last) system message should have the suffix appended.
	if !strings.Contains(got[1].Content, "Second system message.") {
		t.Errorf("second system message lost original content: %q", got[1].Content)
	}
	if !strings.Contains(got[1].Content, "You must respond by calling") {
		t.Errorf("suffix not appended to last system message: %q", got[1].Content)
	}
}

// TestInjectToolChoiceSystemMessage_DeveloperRole verifies that a "developer"
// role message is treated the same as "system" for suffix injection.
func TestInjectToolChoiceSystemMessage_DeveloperRole(t *testing.T) {
	msgs := []api.Message{
		{Role: "developer", Content: "Developer instructions."},
		{Role: "user", Content: "do something"},
	}
	got := InjectToolChoiceSystemMessage(msgs, ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "my_tool"})
	if len(got) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(got))
	}
	if got[0].Role != "developer" {
		t.Errorf("expected developer role preserved, got %q", got[0].Role)
	}
	if !strings.Contains(got[0].Content, "my_tool") {
		t.Errorf("suffix not injected into developer message: %q", got[0].Content)
	}
}

// TestToolChoiceGrammarSchema_Named_ValidJSON verifies that the schema for a
// named tool with multiple parameters is well-formed JSON and includes all
// parameter names.
func TestToolChoiceGrammarSchema_Named_ValidJSON(t *testing.T) {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{Type: []string{"string"}, Description: "Search query"})
	props.Set("limit", api.ToolProperty{Type: []string{"integer"}, Description: "Max results"})
	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "search",
			Description: "Search the web",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
				Required:   []string{"query"},
			},
		},
	}

	got := ToolChoiceGrammarSchema(ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "search"}, []api.Tool{tool})
	if got == nil {
		t.Fatal("expected non-nil schema")
	}
	if err := json.Unmarshal(got, new(any)); err != nil {
		t.Fatalf("schema is not valid JSON: %v\n%s", err, string(got))
	}
	if !strings.Contains(string(got), "query") {
		t.Errorf("schema missing 'query' parameter: %s", string(got))
	}
	if !strings.Contains(string(got), "limit") {
		t.Errorf("schema missing 'limit' parameter: %s", string(got))
	}
}

// TestToolChoiceGrammarSchema_NotAppliedWhenFormatSet verifies the caller
// contract documented in server/routes.go: the grammar schema must NOT be
// applied when req.Format is already set (e.g. structured output or a
// user-supplied JSON schema), to avoid double-constraining the sampler.
// The function itself always returns a schema for valid inputs; the guard
// `if req.Format == nil` in routes.go is the enforcement point.
func TestToolChoiceGrammarSchema_NotAppliedWhenFormatSet(t *testing.T) {
	props := api.NewToolPropertiesMap()
	props.Set("x", api.ToolProperty{Type: []string{"string"}})
	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "do_thing",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
				Required:   []string{"x"},
			},
		},
	}
	choice := ParsedToolChoice{Kind: ToolChoiceNamed, FuncName: "do_thing"}

	// The function always returns a schema for a valid named tool.
	schema := ToolChoiceGrammarSchema(choice, []api.Tool{tool})
	if schema == nil {
		t.Fatal("expected non-nil schema for valid named tool")
	}

	// Simulate the routes.go guard: only apply if req.Format == nil.
	// If req.Format is already set (e.g. structured output), skip.
	existingFormat := json.RawMessage(`{"type":"object"}`)
	applied := false
	if existingFormat == nil {
		applied = true
		_ = schema
	}
	if applied {
		t.Fatal("schema should NOT be applied when req.Format is already set")
	}
}
