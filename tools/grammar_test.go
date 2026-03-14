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
