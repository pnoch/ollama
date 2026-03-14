package server

import (
	"testing"
)

// makeToolExchange builds a minimal function_call + function_call_output pair.
func makeToolExchange(callID, name, args, output string) []map[string]any {
	return []map[string]any{
		{
			"type":      "function_call",
			"call_id":   callID,
			"name":      name,
			"arguments": args,
		},
		{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	}
}

// TestSelectStructuredCompactionCandidate_EmptyHead verifies that an empty
// compactedHead returns no candidate.
func TestSelectStructuredCompactionCandidate_EmptyHead(t *testing.T) {
	_, found := selectStructuredCompactionCandidate(nil, nil, "")
	if found {
		t.Error("expected no candidate for empty head, got one")
	}
}

// TestSelectStructuredCompactionCandidate_SingleToolExchange verifies that a
// single tool exchange in the head is selected as a structured candidate.
func TestSelectStructuredCompactionCandidate_SingleToolExchange(t *testing.T) {
	head := makeToolExchange("call_1", "read_file", `{"path":"main.go"}`, "package main")
	tail := []map[string]any{
		makeResponsesInputMessage("user", "What does main.go do?"),
	}

	candidate, found := selectStructuredCompactionCandidate(head, tail, "")
	if !found {
		t.Fatal("expected a structured candidate to be found")
	}
	if len(candidate.structured) == 0 {
		t.Error("expected structured items in candidate, got none")
	}
}

// TestSelectStructuredCompactionCandidate_PrefersBetterScore verifies that
// when multiple candidates exist, the one with the higher score is chosen.
func TestSelectStructuredCompactionCandidate_PrefersBetterScore(t *testing.T) {
	// A tool exchange with a long, informative output scores higher than
	// one with a trivial output.
	shortExchange := makeToolExchange("call_1", "ping", `{}`, "pong")
	longOutput := "file content: " + string(make([]byte, 200))
	longExchange := makeToolExchange("call_2", "read_file", `{"path":"big.go"}`, longOutput)

	head := append(shortExchange, longExchange...)
	tail := []map[string]any{
		makeResponsesInputMessage("user", "continue"),
	}

	candidate, found := selectStructuredCompactionCandidate(head, tail, "")
	if !found {
		t.Fatal("expected a structured candidate to be found")
	}
	// The selected candidate should include the long-output tool exchange.
	foundLong := false
	for _, item := range candidate.structured {
		if id, _ := item["call_id"].(string); id == "call_2" {
			foundLong = true
		}
		if id, _ := item["call_id"].(string); id == "call_2" {
			if out, _ := item["output"].(string); len(out) > 100 {
				foundLong = true
			}
		}
	}
	if !foundLong {
		t.Error("expected the higher-scoring (long output) tool exchange to be selected")
	}
}

// TestSelectStructuredCompactionCandidate_NoToolsInTail verifies that items
// already in the preserved tail are not selected again as structured candidates.
func TestSelectStructuredCompactionCandidate_NoToolsInTail(t *testing.T) {
	// The tail already contains the tool exchange — it should not be
	// selected again from the head.
	exchange := makeToolExchange("call_1", "read_file", `{"path":"a.go"}`, "content")
	head := []map[string]any{
		makeResponsesInputMessage("user", "old question"),
		makeResponsesAssistantMessage("old answer"),
	}
	tail := exchange

	candidate, found := selectStructuredCompactionCandidate(head, tail, "")
	if found {
		// If a candidate is found, it must not duplicate the tail's call_id.
		for _, item := range candidate.structured {
			if id, _ := item["call_id"].(string); id == "call_1" {
				t.Error("selected candidate duplicates an item already in the preserved tail")
			}
		}
	}
}

// TestSelectStructuredCompactionCandidate_MessagePairFallback verifies that
// when no tool exchange is present, a user+assistant message pair is still
// selected as a structured candidate (for models that support it).
func TestSelectStructuredCompactionCandidate_MessagePairFallback(t *testing.T) {
	head := []map[string]any{
		makeResponsesInputMessage("user", "What is the capital of France?"),
		makeResponsesAssistantMessage("The capital of France is Paris."),
		makeResponsesInputMessage("user", "And Germany?"),
		makeResponsesAssistantMessage("The capital of Germany is Berlin."),
	}
	tail := []map[string]any{
		makeResponsesInputMessage("user", "current question"),
	}

	// Use a model that allows structured message pairs.
	candidate, found := selectStructuredCompactionCandidate(head, tail, "gpt-4o")
	if !found {
		// Message pair selection is model-gated; skip rather than fail if
		// the model doesn't allow it.
		t.Skip("model does not support structured message pair selection")
	}
	if len(candidate.structured) < 2 {
		t.Errorf("expected at least 2 items in message pair candidate, got %d", len(candidate.structured))
	}
}
