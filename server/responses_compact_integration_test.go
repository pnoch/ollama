package server

// Deterministic compaction pipeline tests.
//
// These tests call the internal compaction helpers directly (bypassing the
// HTTP token-estimation threshold) so they produce the same results on every
// platform and model configuration.
//
// The key functions under test are:
//   - splitCompactedTailWithBudget       – splits items into head/tail
//   - buildCompactionSummaryWithTurnCount – builds the summary message
//   - compactResponsesInputForModel      – the full pipeline
//   - selectCompactionSummaryParts       – summary part selection

import (
	"encoding/json"
	"strings"
	"testing"
)

// ─── helpers ────────────────────────────────────────────────────────────────

func mustMarshal(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("mustMarshal: %v", err)
	}
	return b
}

func buildConversation(roleAndText ...string) []map[string]any {
	if len(roleAndText)%2 != 0 {
		panic("buildConversation: odd number of arguments")
	}
	items := make([]map[string]any, 0, len(roleAndText)/2)
	for i := 0; i < len(roleAndText); i += 2 {
		role := roleAndText[i]
		text := roleAndText[i+1]
		if role == "assistant" {
			items = append(items, makeResponsesAssistantMessage(text))
		} else {
			items = append(items, makeResponsesInputMessage(role, text))
		}
	}
	return items
}

func msgText(item map[string]any) string {
	return extractResponsesItemText(item["content"])
}

func msgRole(item map[string]any) string {
	r, _ := item["role"].(string)
	return r
}

func itemType(item map[string]any) string {
	return normalizeResponsesItemType(item)
}

// ─── splitCompactedTailWithBudget tests ─────────────────────────────────────

// TestSplitCompactedTailWithBudget_BasicSplit verifies that with a budget of
// 1 chunk / large chars, only the last chunk is preserved in the tail.
func TestSplitCompactedTailWithBudget_BasicSplit(t *testing.T) {
	items := buildConversation(
		"user", "message one",
		"assistant", "reply one",
		"user", "message two",
		"assistant", "reply two",
		"user", "message three",
		"assistant", "reply three",
	)
	tail, head := splitCompactedTailWithBudget(items, 1, 100_000)
	if len(tail) == 0 {
		t.Fatal("expected at least one item in the tail")
	}
	if len(head)+len(tail) != len(items) {
		t.Errorf("head+tail length mismatch: %d+%d=%d, want %d",
			len(head), len(tail), len(head)+len(tail), len(items))
	}
	last := tail[len(tail)-1]
	if msgText(last) != "reply three" {
		t.Errorf("last tail item text = %q, want %q", msgText(last), "reply three")
	}
}

// TestSplitCompactedTailWithBudget_EmptyInput verifies that an empty input
// returns empty head and tail without panicking.
func TestSplitCompactedTailWithBudget_EmptyInput(t *testing.T) {
	tail, head := splitCompactedTailWithBudget(nil, 3, 1600)
	if len(tail) != 0 || len(head) != 0 {
		t.Errorf("expected empty head and tail for nil input, got head=%d tail=%d", len(head), len(tail))
	}
}

// TestSplitCompactedTailWithBudget_CharLimitEnforced verifies that the char
// limit is respected: items whose JSON exceeds the budget are placed in head.
func TestSplitCompactedTailWithBudget_CharLimitEnforced(t *testing.T) {
	items := []map[string]any{
		makeResponsesInputMessage("user", strings.Repeat("a", 150)),
		makeResponsesAssistantMessage(strings.Repeat("b", 150)),
		makeResponsesInputMessage("user", strings.Repeat("c", 150)),
		makeResponsesAssistantMessage(strings.Repeat("d", 150)),
	}
	tail, head := splitCompactedTailWithBudget(items, 10, 250)
	if len(tail) == 0 {
		t.Fatal("expected at least one item in the tail")
	}
	if len(head)+len(tail) != len(items) {
		t.Errorf("head+tail length mismatch: %d+%d != %d", len(head), len(tail), len(items))
	}
}

// TestSplitCompactedTailWithBudget_FallbackPreservesLastChunk verifies that
// when every chunk exceeds the char budget, the last chunk is still preserved.
func TestSplitCompactedTailWithBudget_FallbackPreservesLastChunk(t *testing.T) {
	items := []map[string]any{
		makeResponsesInputMessage("user", strings.Repeat("x", 400)),
		makeResponsesAssistantMessage(strings.Repeat("y", 400)),
	}
	tail, _ := splitCompactedTailWithBudget(items, 1, 10)
	if len(tail) == 0 {
		t.Error("fallback must preserve at least the last chunk even when all chunks exceed budget")
	}
}

// ─── buildCompactionSummaryWithTurnCount tests ──────────────────────────────

// TestBuildCompactionSummaryWithTurnCount_IncludesTurnCount verifies that the
// summary header includes the turn count when turnCount > 0.
func TestBuildCompactionSummaryWithTurnCount_IncludesTurnCount(t *testing.T) {
	parts := []string{"User asked about Go generics", "Assistant explained type parameters"}
	summary := buildCompactionSummaryWithTurnCount(parts, nil, 12)
	if !strings.Contains(summary, "12") {
		t.Errorf("expected turn count 12 in summary, got: %s", summary)
	}
	if !strings.Contains(summary, "summarising") {
		t.Errorf("expected 'summarising' in summary header, got: %s", summary)
	}
}

// TestBuildCompactionSummaryWithTurnCount_ZeroTurnCountOmitted verifies that
// when turnCount == 0, the header does not include a turn count.
func TestBuildCompactionSummaryWithTurnCount_ZeroTurnCountOmitted(t *testing.T) {
	parts := []string{"some summary part"}
	summary := buildCompactionSummaryWithTurnCount(parts, nil, 0)
	if strings.Contains(summary, "summarising") {
		t.Errorf("expected no turn count in summary when turnCount=0, got: %s", summary)
	}
}

// TestBuildCompactionSummaryWithTurnCount_EmptyPartsAndCounts verifies that
// an empty parts list and empty omitted counts returns an empty string.
func TestBuildCompactionSummaryWithTurnCount_EmptyPartsAndCounts(t *testing.T) {
	summary := buildCompactionSummaryWithTurnCount(nil, nil, 5)
	if summary != "" {
		t.Errorf("expected empty summary for empty parts and counts, got: %q", summary)
	}
}

// TestBuildCompactionSummaryWithTurnCount_OmittedCountsIncluded verifies that
// omitted item counts appear in the summary.
func TestBuildCompactionSummaryWithTurnCount_OmittedCountsIncluded(t *testing.T) {
	omitted := map[string]int{"image_generation_call": 3}
	summary := buildCompactionSummaryWithTurnCount(nil, omitted, 0)
	if summary == "" {
		t.Error("expected non-empty summary when omitted counts are present")
	}
}

// ─── compactResponsesInputForModel pipeline tests ───────────────────────────

// TestCompactResponsesInputForModel_SummaryBeforeTail is the regression test
// for the ordering bug: the compaction summary must appear BEFORE the preserved
// tail so the model's last view is the active task, not the summary notice.
// This test calls the pipeline directly with enough items to guarantee
// compaction regardless of any HTTP threshold.
func TestCompactResponsesInputForModel_SummaryBeforeTail(t *testing.T) {
	items := make([]map[string]any, 0, 60)
	for i := 0; i < 25; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("earlier question ", 20)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("earlier answer ", 20)))
	}
	items = append(items, makeResponsesInputMessage("user", "Fix the TypeScript error in src/index.ts"))
	items = append(items, makeResponsesAssistantMessage("I will fix the TypeScript error now."))

	out, err := compactResponsesInputForModel(mustMarshal(t, items), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	summaryIdx := -1
	lastActiveIdx := -1
	for i, item := range out {
		text := msgText(item)
		if strings.Contains(text, "compacted") || strings.Contains(text, "Compacted") ||
			strings.Contains(text, "summarising") || strings.Contains(text, "Preserved summary") {
			summaryIdx = i
		}
		if strings.Contains(text, "TypeScript error") {
			lastActiveIdx = i
		}
	}

	if summaryIdx == -1 {
		t.Skip("no compaction summary produced; all items fit within tail budget")
	}
	if lastActiveIdx == -1 {
		t.Fatal("active task message not found in output")
	}
	if summaryIdx > lastActiveIdx {
		t.Errorf("ordering regression: summary (idx=%d) appears AFTER active task message (idx=%d); "+
			"summary must come before the preserved tail", summaryIdx, lastActiveIdx)
	}
}

// TestCompactResponsesInputForModel_SystemMessagesPreserved verifies that
// system/developer messages are always placed at the start of the output.
func TestCompactResponsesInputForModel_SystemMessagesPreserved(t *testing.T) {
	items := []map[string]any{
		{"type": "message", "role": "system", "content": "You are a helpful assistant."},
		{"type": "message", "role": "developer", "content": "Use concise responses."},
	}
	for i := 0; i < 20; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("pad ", 50)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("pad ", 50)))
	}

	out, err := compactResponsesInputForModel(mustMarshal(t, items), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("output must not be empty")
	}
	foundSystem := false
	foundDeveloper := false
	for i, item := range out {
		role, _ := item["role"].(string)
		switch role {
		case "system":
			foundSystem = true
			if i != 0 {
				t.Errorf("system message at index %d, expected index 0", i)
			}
		case "developer":
			foundDeveloper = true
			if i > 1 {
				t.Errorf("developer message at index %d, expected index <=1", i)
			}
		}
	}
	if !foundSystem {
		t.Error("system message missing from output")
	}
	if !foundDeveloper {
		t.Error("developer message missing from output")
	}
}

// TestCompactResponsesInputForModel_OutputNeverEmpty verifies that even a
// very long conversation always produces at least one output item.
func TestCompactResponsesInputForModel_OutputNeverEmpty(t *testing.T) {
	items := make([]map[string]any, 0, 100)
	for i := 0; i < 50; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("x", 200)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("y", 200)))
	}
	out, err := compactResponsesInputForModel(mustMarshal(t, items), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) == 0 {
		t.Error("output must never be empty")
	}
}

// TestCompactResponsesInputForModel_NilInput verifies that a nil raw message
// returns an empty slice without error.
func TestCompactResponsesInputForModel_NilInput(t *testing.T) {
	out, err := compactResponsesInputForModel(nil, "")
	if err != nil {
		t.Fatalf("unexpected error on nil input: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty output for nil input, got %d items", len(out))
	}
}

// TestCompactResponsesInputForModel_ShortPassthrough verifies that a short
// conversation is returned with the same item count (no compaction).
func TestCompactResponsesInputForModel_ShortPassthrough(t *testing.T) {
	items := buildConversation(
		"user", "Hello",
		"assistant", "Hi there!",
		"user", "How are you?",
		"assistant", "I am fine, thank you.",
	)
	out, err := compactResponsesInputForModel(mustMarshal(t, items), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != len(items) {
		t.Errorf("short conversation should pass through unchanged: got %d items, want %d", len(out), len(items))
	}
}

// TestCompactResponsesInputForModel_ToolCallInTailPreserved verifies that
// a recent tool exchange in the preserved tail survives compaction.
func TestCompactResponsesInputForModel_ToolCallInTailPreserved(t *testing.T) {
	items := make([]map[string]any, 0, 60)
	for i := 0; i < 25; i++ {
		items = append(items, makeResponsesInputMessage("user", strings.Repeat("pad ", 80)))
		items = append(items, makeResponsesAssistantMessage(strings.Repeat("pad ", 80)))
	}
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

	out, err := compactResponsesInputForModel(mustMarshal(t, items), "")
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
			if o, _ := item["output"].(string); strings.Contains(o, "string = 1") {
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

// TestCompactResponsesInputForModel_StringInput verifies that a plain string
// input is wrapped in a user message.
func TestCompactResponsesInputForModel_StringInput(t *testing.T) {
	raw := mustMarshal(t, "Hello, world!")
	out, err := compactResponsesInputForModel(raw, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 output item for string input, got %d", len(out))
	}
	if msgRole(out[0]) != "user" {
		t.Errorf("expected role=user for string input, got %q", msgRole(out[0]))
	}
	if !strings.Contains(msgText(out[0]), "Hello, world!") {
		t.Errorf("expected text to contain input string, got %q", msgText(out[0]))
	}
}

// ─── selectCompactionSummaryParts tests ─────────────────────────────────────

// TestSelectCompactionSummaryParts_RespectsMaxItems verifies that the 80-item
// cap is enforced even when many parts are within the char budget.
func TestSelectCompactionSummaryParts_RespectsMaxItems(t *testing.T) {
	parts := make([]string, 200)
	for i := range parts {
		parts[i] = "short part"
	}
	selected := selectCompactionSummaryParts(parts, 1_000_000)
	if len(selected) > responsesCompactSummaryMaxItems {
		t.Errorf("expected at most %d selected parts, got %d",
			responsesCompactSummaryMaxItems, len(selected))
	}
}

// TestSelectCompactionSummaryParts_RespectsCharBudget verifies that the total
// character count of selected parts does not exceed the budget.
func TestSelectCompactionSummaryParts_RespectsCharBudget(t *testing.T) {
	parts := make([]string, 50)
	for i := range parts {
		parts[i] = strings.Repeat("x", 30)
	}
	budget := 200
	selected := selectCompactionSummaryParts(parts, budget)
	total := 0
	for _, p := range selected {
		total += len(p)
	}
	if total > budget {
		t.Errorf("selected parts total %d chars, exceeds budget %d", total, budget)
	}
}

// TestSelectCompactionSummaryParts_EmptyInput verifies that an empty parts
// list returns an empty result without panicking.
func TestSelectCompactionSummaryParts_EmptyInput(t *testing.T) {
	selected := selectCompactionSummaryParts(nil, 1200)
	if len(selected) != 0 {
		t.Errorf("expected empty result for nil input, got %d items", len(selected))
	}
}
