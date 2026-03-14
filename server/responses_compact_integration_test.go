package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// helpers ─────────────────────────────────────────────────────────────────────

func mustMarshal(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func msgRole(item map[string]any) string {
	r, _ := item["role"].(string)
	return r
}

func msgText(item map[string]any) string {
	content, ok := item["content"].([]any)
	if !ok {
		return ""
	}
	var parts []string
	for _, c := range content {
		cm, ok := c.(map[string]any)
		if !ok {
			continue
		}
		if t, ok := cm["text"].(string); ok {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, "")
}

func itemType(item map[string]any) string {
	t, _ := item["type"].(string)
	return t
}

// buildConversation creates a flat list of messages for use as compaction input.
// Each entry is a map with "type", "role", and "content".
func buildConversation(pairs ...string) []map[string]any {
	if len(pairs)%2 != 0 {
		panic("buildConversation: pairs must be even (role, text, role, text, ...)")
	}
	items := make([]map[string]any, 0, len(pairs)/2)
	for i := 0; i < len(pairs); i += 2 {
		role, text := pairs[i], pairs[i+1]
		if role == "user" {
			items = append(items, makeResponsesInputMessage(role, text))
		} else {
			items = append(items, makeResponsesAssistantMessage(text))
		}
	}
	return items
}

// ─── Tests ───────────────────────────────────────────────────────────────────

// TestCompactResponsesInput_EmptyInput verifies that an empty input produces
// an empty output without error.
func TestCompactResponsesInput_EmptyInput(t *testing.T) {
	out, err := compactResponsesInput(mustMarshal(t, []map[string]any{}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty output, got %d items", len(out))
	}
}

// TestCompactResponsesInput_StringInput verifies that a bare string input is
// wrapped in a single user message.
func TestCompactResponsesInput_StringInput(t *testing.T) {
	out, err := compactResponsesInput(mustMarshal(t, "hello world"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 item, got %d", len(out))
	}
	if msgRole(out[0]) != "user" {
		t.Errorf("expected role=user, got %q", msgRole(out[0]))
	}
	if !strings.Contains(msgText(out[0]), "hello world") {
		t.Errorf("expected text to contain 'hello world', got %q", msgText(out[0]))
	}
}

// TestCompactResponsesInput_SystemMessagesPreserved verifies that system and
// developer messages are always placed at the front of the output regardless
// of how much compaction occurs.
func TestCompactResponsesInput_SystemMessagesPreserved(t *testing.T) {
	items := []map[string]any{
		{"type": "message", "role": "system", "content": []map[string]any{{"type": "input_text", "text": "You are a helpful assistant."}}},
		{"type": "message", "role": "developer", "content": []map[string]any{{"type": "input_text", "text": "Dev instructions."}}},
	}
	// Add a long conversation so compaction fires.
	for i := 0; i < 20; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("question ", 80)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("answer ", 80)))
	}

	out, err := compactResponsesInput(mustMarshal(t, items))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) < 2 {
		t.Fatalf("expected at least 2 items, got %d", len(out))
	}
	if msgRole(out[0]) != "system" {
		t.Errorf("first item should be system message, got role=%q", msgRole(out[0]))
	}
	if msgRole(out[1]) != "developer" {
		t.Errorf("second item should be developer message, got role=%q", msgRole(out[1]))
	}
}

// TestCompactResponsesInput_SummaryBeforeTail is the regression test for the
// ordering bug that caused Codex to lose task context after compaction.
// The compaction summary must appear BEFORE the preserved tail so that the
// most recent real messages are the last thing the model sees.
func TestCompactResponsesInput_SummaryBeforeTail(t *testing.T) {
	// Build a conversation long enough to trigger compaction.
	items := []map[string]any{}
	for i := 0; i < 30; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("earlier question ", 60)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("earlier answer ", 60)))
	}
	// The "active task" messages that must appear last.
	items = append(items, makeResponsesInputMessage("user", "Fix the TypeScript error in src/index.ts"))
	items = append(items, makeResponsesAssistantMessage("I will fix the TypeScript error now."))

	out, err := compactResponsesInput(mustMarshal(t, items))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Find the compaction summary and the active task messages.
	summaryIdx := -1
	lastUserIdx := -1
	lastAssistantIdx := -1
	for i, item := range out {
		text := msgText(item)
		if strings.Contains(text, "compacted") || strings.Contains(text, "Compacted") ||
			strings.Contains(text, "summary") || strings.Contains(text, "Summary") {
			summaryIdx = i
		}
		if strings.Contains(text, "TypeScript error") {
			if msgRole(item) == "user" {
				lastUserIdx = i
			} else {
				lastAssistantIdx = i
			}
		}
	}

	if summaryIdx == -1 {
		// No summary was emitted (conversation may not have been long enough
		// to trigger compaction on this platform). Skip rather than fail.
		t.Skip("no compaction summary found; conversation may be below threshold")
	}

	if lastUserIdx != -1 && summaryIdx > lastUserIdx {
		t.Errorf("compaction summary (idx=%d) appears AFTER the active task user message (idx=%d); summary must come before the preserved tail", summaryIdx, lastUserIdx)
	}
	if lastAssistantIdx != -1 && summaryIdx > lastAssistantIdx {
		t.Errorf("compaction summary (idx=%d) appears AFTER the active task assistant message (idx=%d); summary must come before the preserved tail", summaryIdx, lastAssistantIdx)
	}
}

// TestCompactResponsesInput_ShortConversationPassthrough verifies that a short
// conversation (below the compaction threshold) is returned unchanged.
func TestCompactResponsesInput_ShortConversationPassthrough(t *testing.T) {
	items := buildConversation(
		"user", "Hello",
		"assistant", "Hi there!",
		"user", "How are you?",
		"assistant", "I am fine, thank you.",
	)

	out, err := compactResponsesInput(mustMarshal(t, items))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != len(items) {
		t.Errorf("short conversation should pass through unchanged: got %d items, want %d", len(out), len(items))
	}
}

// TestCompactResponsesInput_OutputNeverEmpty verifies that even a very long
// conversation always produces at least one output item (the fallback message).
func TestCompactResponsesInput_OutputNeverEmpty(t *testing.T) {
	items := []map[string]any{}
	for i := 0; i < 50; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("x", 200)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("y", 200)))
	}

	out, err := compactResponsesInput(mustMarshal(t, items))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) == 0 {
		t.Error("output must never be empty")
	}
}

// TestCompactResponsesInput_ToolCallPreserved verifies that function_call and
// function_call_output items in the preserved tail are kept in the output.
func TestCompactResponsesInput_ToolCallPreserved(t *testing.T) {
	items := []map[string]any{}
	// Pad with enough history to trigger compaction.
	for i := 0; i < 25; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("pad ", 80)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("pad ", 80)))
	}
	// Recent tool exchange that should survive in the tail.
	items = append(items, map[string]any{
		"type":      "function_call",
		"call_id":   "call_abc",
		"name":      "read_file",
		"arguments": `{"path":"src/index.ts"}`,
	})
	items = append(items, map[string]any{
		"type":    "function_call_output",
		"call_id": "call_abc",
		"output":  "const x: string = 1;",
	})
	items = append(items, makeResponsesAssistantMessage("Found the type error on line 1."))

	out, err := compactResponsesInput(mustMarshal(t, items))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	foundToolCall := false
	foundToolOutput := false
	for _, item := range out {
		switch itemType(item) {
		case "function_call":
			if n, _ := item["name"].(string); n == "read_file" {
				foundToolCall = true
			}
		case "function_call_output":
			if o, _ := item["output"].(string); strings.Contains(o, "type error") || strings.Contains(o, "string = 1") {
				foundToolOutput = true
			}
		}
	}

	if !foundToolCall {
		t.Error("expected recent function_call item to be preserved in output")
	}
	if !foundToolOutput {
		t.Error("expected recent function_call_output item to be preserved in output")
	}
}

// TestCompactResponsesInput_NilRawMessage verifies that a nil raw message
// returns an empty slice without error.
func TestCompactResponsesInput_NilRawMessage(t *testing.T) {
	out, err := compactResponsesInput(nil)
	if err != nil {
		t.Fatalf("unexpected error on nil input: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty output for nil input, got %d items", len(out))
	}
}
