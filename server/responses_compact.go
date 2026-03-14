package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/ollama/ollama/internal/strutil"
)

const (
	responsesCompactRecentTailMaxChunks = 3
	responsesCompactRecentTailMaxChars  = 1600
	responsesCompactSummaryMaxChars     = 1200
	responsesCompactSameRoleRunMaxItems = 3
	responsesCompactSameRoleRunMaxChars = 180
)

var referentialListLinePattern = regexp.MustCompile(`(?m)^\s*(?:[-*]|\d+[.)])\s+\S+`)

func (s *Server) ResponsesCompactHandler(c *gin.Context) {
	body, err := readRequestBody(c.Request)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelName, _ := extractModelField(body)

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	output, err := compactResponsesInput(payload["input"])
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":         fmt.Sprintf("respcomp_%s", uuid.New().String()),
		"object":     "response.compaction",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      modelName,
		"output":     output,
		"usage": gin.H{
			"input_tokens":  0,
			"output_tokens": 0,
			"total_tokens":  0,
			"input_tokens_details": gin.H{
				"cached_tokens": 0,
			},
			"output_tokens_details": gin.H{
				"reasoning_tokens": 0,
			},
		},
	})
}

func compactResponsesInput(raw json.RawMessage) ([]map[string]any, error) {
	return compactResponsesInputForModel(raw, "")
}

func compactResponsesInputForModel(raw json.RawMessage, model string) ([]map[string]any, error) {
	if len(raw) == 0 {
		return []map[string]any{}, nil
	}

	var inputText string
	if err := json.Unmarshal(raw, &inputText); err == nil {
		return []map[string]any{
			makeResponsesInputMessage("user", inputText),
		}, nil
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, err
	}

	systemAndDeveloper := make([]map[string]any, 0, len(items))
	otherItems := make([]map[string]any, 0, len(items))
	summaryParts := make([]string, 0, len(items))
	omittedCounts := map[string]int{}
	structuredSelections := make([]structuredCompactionCandidate, 0, 2)

	for _, item := range items {
		itemType := normalizeResponsesItemType(item)
		role, _ := item["role"].(string)

		if itemType == "message" && slices.Contains([]string{"system", "developer"}, role) {
			systemAndDeveloper = append(systemAndDeveloper, item)
			continue
		}

		otherItems = append(otherItems, item)
	}

	chunkMax, charMax := responsesCompactTailBudget(model)
	preservedTail, compactedHead := splitCompactedTailWithBudget(otherItems, chunkMax, charMax)
	for n := 0; n < responsesStructuredChunkBudget(model); n++ {
		selected, ok := selectStructuredCompactionCandidate(compactedHead, preservedTail, model)
		if !ok {
			break
		}
		structuredSelections = append(structuredSelections, selected)
		if selected.tailDrop > 0 {
			preservedTail = preservedTail[selected.tailDrop:]
		}
		if selected.headStart >= 0 && selected.headEnd >= selected.headStart {
			compactedHead = append(
				append([]map[string]any{}, compactedHead[:selected.headStart]...),
				compactedHead[selected.headEnd+1:]...,
			)
		}
	}
	for i := 0; i < len(compactedHead); i++ {
		item := compactedHead[i]
		itemType := normalizeResponsesItemType(item)
		if combined, nextIndex, ok := summarizeCompactedUserToolAssistantRun(compactedHead, i); ok {
			summaryParts = append(summaryParts, combined)
			i = nextIndex
			continue
		}
		if combined, nextIndex, ok := summarizeCompactedToolExchangeWithAssistant(compactedHead, i); ok {
			summaryParts = append(summaryParts, combined)
			i = nextIndex
			continue
		}
		if combined, nextIndex, ok := summarizeCompactedSameRoleMessageRun(compactedHead, i); ok {
			summaryParts = append(summaryParts, combined)
			i = nextIndex
			continue
		}
		if combined, nextIndex, ok := summarizeCompactedMessageRun(compactedHead, i); ok {
			if !summaryDuplicatesPreserved(combined, preservedTail) {
				summaryParts = append(summaryParts, combined)
			}
			i = nextIndex
			continue
		}
		if combined, nextIndex, ok := summarizeCompactedToolExchange(compactedHead, i); ok {
			summaryParts = append(summaryParts, combined)
			i = nextIndex
			continue
		}
		if summary := summarizeCompactedInputItem(itemType, item); summary != "" {
			summaryParts = append(summaryParts, summary)
		} else {
			omittedCounts[itemType]++
		}
	}

	output := make([]map[string]any, 0, len(systemAndDeveloper)+len(preservedTail)+1)
	output = append(output, systemAndDeveloper...)
	structuredPreserved := make([]map[string]any, 0, 8)
	if len(structuredSelections) > 0 {
		slices.SortFunc(structuredSelections, func(a, b structuredCompactionCandidate) int {
			if a.headStart < b.headStart {
				return -1
			}
			if a.headStart > b.headStart {
				return 1
			}
			return 0
		})
		for _, selection := range structuredSelections {
			for _, item := range selection.structured {
				if shouldSkipStructuredPreservedItem(structuredPreserved, preservedTail, item) {
					continue
				}
				structuredPreserved = append(structuredPreserved, item)
			}
		}
	}
	output = append(output, structuredPreserved...)
	// Count turns as the number of user messages in compactedHead.
	// Using len(compactedHead) would over-count because each turn can span
	// multiple items (function_call, function_call_output, assistant message).
	turnCount := 0
	for _, item := range compactedHead {
		if normalizeResponsesItemType(item) == "message" {
			if role, _ := item["role"].(string); role == "user" {
				turnCount++
			}
		}
	}
	// Insert the compaction summary BEFORE preservedTail so that the most
	// recent real messages remain the last thing the model sees. Placing the
	// summary after preservedTail caused the model to treat the summary as
	// the latest assistant turn and lose track of the active task.
	if summaryText := buildCompactionSummaryWithTurnCount(summaryParts, omittedCounts, turnCount); summaryText != "" {
		output = append(output, makeResponsesAssistantMessage(summaryText))
	}
	output = append(output, preservedTail...)
	if len(output) == 0 {
		output = append(output, makeResponsesAssistantMessage("Previous conversation history was compacted by Ollama."))
	}

	return output, nil
}

func splitCompactedTail(items []map[string]any) (preservedTail []map[string]any, compactedHead []map[string]any) {
	chunkMax, charMax := responsesCompactTailBudget("")
	return splitCompactedTailWithBudget(items, chunkMax, charMax)
}

func splitCompactedTailWithBudget(items []map[string]any, chunkMax, charMax int) (preservedTail []map[string]any, compactedHead []map[string]any) {
	if len(items) == 0 {
		return nil, nil
	}

	chunks := buildResponsesCompactionChunks(items)
	startChunk := len(chunks)
	size := 0
	for startChunk > 0 {
		candidate := chunks[startChunk-1]
		candidateJSON, err := json.Marshal(candidate)
		if err != nil {
			slog.Warn("responses compaction: failed to marshal chunk during tail expansion", "error", err)
			break
		}
		if len(chunks)-startChunk >= chunkMax {
			break
		}
		if size+len(candidateJSON) > charMax {
			break
		}
		startChunk--
		size += len(candidateJSON)
	}
	// If the loop exited without preserving anything (e.g. marshal error on the
	// very first candidate), fall back to preserving at least the last chunk so
	// we never compact the entire conversation with nothing preserved.
	if startChunk == len(chunks) && len(chunks) > 0 {
		startChunk = len(chunks) - 1
	}
	startItem := 0
	for i := 0; i < startChunk; i++ {
		startItem += len(chunks[i])
	}

	preservedTail = items[startItem:]
	compactedHead = items[:startItem]
	return preserveLatestAssistantReferentialAnchor(compactedHead, preservedTail)
}

func responsesCompactTailBudget(model string) (chunkMax, charMax int) {
	// Default tail budget for unknown or small-context models.
	chunkMax = responsesCompactRecentTailMaxChunks
	charMax = responsesCompactRecentTailMaxChars

	limit, ok := lookupCloudModelLimit(model)
	if !ok || limit.Context <= 0 {
		return chunkMax, charMax
	}

	// Scale the char budget to ~0.8% of the model context window (in chars,
	// approximating 3 chars/token). This gives larger-context models a
	// proportionally bigger preserved tail without wasting context on small
	// models.
	//
	// Examples:
	//   32k  context → ~768  chars  (≈ 256 tokens)
	//   128k context → ~3072 chars  (≈ 1024 tokens)
	//   200k context → ~4800 chars  (≈ 1600 tokens)
	//   262k context → ~6291 chars  (≈ 2097 tokens)
	scaledChars := limit.Context * 3 / 400 // 0.75% of context in chars
	if scaledChars < responsesCompactRecentTailMaxChars {
		scaledChars = responsesCompactRecentTailMaxChars
	}
	const maxTailChars = 8000
	if scaledChars > maxTailChars {
		scaledChars = maxTailChars
	}

	// Scale chunks: 1 extra chunk per 64k of context beyond the first 32k.
	scaledChunks := responsesCompactRecentTailMaxChunks + (limit.Context-32_000)/64_000
	if scaledChunks < responsesCompactRecentTailMaxChunks {
		scaledChunks = responsesCompactRecentTailMaxChunks
	}
	const maxTailChunks = 8
	if scaledChunks > maxTailChunks {
		scaledChunks = maxTailChunks
	}

	return scaledChunks, scaledChars
}

func responsesStructuredChunkBudget(model string) int {
	// Scale the structured chunk budget proportionally with context window:
	//   < 64k  context → 1 structured chunk
	//   64k–128k       → 2 structured chunks
	//   128k–256k      → 3 structured chunks
	//   256k+          → 4 structured chunks
	// This ensures large-context models can preserve more structured tool
	// exchanges verbatim rather than summarising them.
	if limit, ok := lookupCloudModelLimit(model); ok && limit.Context > 0 {
		switch {
		case limit.Context >= 256_000:
			return 4
		case limit.Context >= 128_000:
			return 3
		case limit.Context >= 64_000:
			return 2
		}
	}
	return 1
}

func shouldSkipStructuredPreservedItem(structuredPreserved, preservedTail []map[string]any, item map[string]any) bool {
	if normalizeResponsesItemType(item) != "message" {
		return false
	}
	role, _ := item["role"].(string)
	if role != "assistant" {
		return false
	}
	text := extractResponsesItemText(item["content"])
	if text == "" {
		return false
	}
	for _, existing := range structuredPreserved {
		if normalizeResponsesItemType(existing) == "message" {
			if existingRole, _ := existing["role"].(string); existingRole == "assistant" && extractResponsesItemText(existing["content"]) == text {
				return true
			}
		}
	}
	for _, existing := range preservedTail {
		if normalizeResponsesItemType(existing) == "message" {
			if existingRole, _ := existing["role"].(string); existingRole == "assistant" && extractResponsesItemText(existing["content"]) == text {
				return true
			}
		}
	}
	return false
}

func preserveLatestAssistantReferentialAnchor(compactedHead, preservedTail []map[string]any) ([]map[string]any, []map[string]any) {
	// Select the anchor with the highest referential list-line count rather than
	// simply the last one. In a long coding session a model may emit multiple
	// file-listing anchors; the most information-dense one is the most useful
	// to preserve regardless of its position in the head.
	bestIdx := -1
	bestCount := 0
	for i, item := range compactedHead {
		count := assistantReferentialAnchorScore(item)
		if count <= 0 {
			continue
		}
		// Prefer higher match count; break ties in favour of the later message
		// (more recent context is generally more relevant).
		if count > bestCount || (count == bestCount && i > bestIdx) {
			bestCount = count
			bestIdx = i
		}
	}
	if bestIdx < 0 {
		return preservedTail, compactedHead
	}
	item := compactedHead[bestIdx]
	newHead := append([]map[string]any{}, compactedHead[:bestIdx]...)
	newHead = append(newHead, compactedHead[bestIdx+1:]...)
	newTail := append([]map[string]any{item}, preservedTail...)
	return newTail, newHead
}

// assistantReferentialAnchorScore returns the number of referential list lines
// in the item's text, or 0 if the item is not a qualifying anchor message.
func assistantReferentialAnchorScore(item map[string]any) int {
	if normalizeResponsesItemType(item) != "message" {
		return 0
	}
	role, _ := item["role"].(string)
	if role != "assistant" {
		return 0
	}
	text := extractResponsesItemRawText(item["content"])
	if text == "" {
		return 0
	}
	matches := referentialListLinePattern.FindAllString(text, -1)
	if len(matches) < 2 {
		return 0
	}
	return len(matches)
}

func isAssistantReferentialAnchorMessage(item map[string]any) bool {
	if normalizeResponsesItemType(item) != "message" {
		return false
	}
	role, _ := item["role"].(string)
	if role != "assistant" {
		return false
	}
	text := extractResponsesItemRawText(item["content"])
	if text == "" {
		return false
	}
	return len(referentialListLinePattern.FindAllString(text, -1)) >= 2
}

func buildResponsesCompactionChunks(items []map[string]any) [][]map[string]any {
	chunks := make([][]map[string]any, 0, len(items))
	for i := 0; i < len(items); i++ {
		item := items[i]
		itemType := normalizeResponsesItemType(item)

		if itemType == "function_call" || itemType == "custom_tool_call" {
			chunk := []map[string]any{item}
			if i+1 < len(items) {
				next := items[i+1]
				nextType := normalizeResponsesItemType(next)
				callID, _ := item["call_id"].(string)
				nextCallID, _ := next["call_id"].(string)
				if (nextType == "function_call_output" || nextType == "custom_tool_call_output") && callID != "" && callID == nextCallID {
					chunk = append(chunk, next)
					i++
				}
			}
			chunks = append(chunks, chunk)
			continue
		}

		chunks = append(chunks, []map[string]any{item})
	}

	return chunks
}

func normalizeResponsesItemType(item map[string]any) string {
	itemType, _ := item["type"].(string)
	if itemType == "" {
		if _, ok := item["role"].(string); ok {
			return "message"
		}
	}
	return itemType
}

func summarizeCompactedInputItem(itemType string, item map[string]any) string {
	switch itemType {
	case "message":
		role, _ := item["role"].(string)
		text := extractResponsesItemText(item["content"])
		if text == "" {
			return ""
		}
		switch role {
		case "assistant":
			return "Assistant: " + text
		case "tool":
			return "Tool output: " + text
		default:
			return role + ": " + text
		}
	case "function_call", "custom_tool_call":
		name, _ := item["name"].(string)
		args := compactSnippet(stringValue(item["arguments"]))
		if args == "" {
			args = compactSnippet(stringValue(item["input"]))
		}
		if args == "" {
			return fmt.Sprintf("Tool call: %s", name)
		}
		return fmt.Sprintf("Tool call %s: %s", name, args)
	case "function_call_output", "custom_tool_call_output":
		output := compactSnippet(stringValue(item["output"]))
		if output == "" {
			return "Tool output recorded."
		}
		return "Tool output: " + output
	case "web_search_call":
		if action, ok := item["action"].(map[string]any); ok {
			if query := strings.TrimSpace(stringValue(action["query"])); query != "" {
				return "Web search: " + compactSnippet(query)
			}
		}
		return "Web search performed."
	case "image_generation_call":
		return "Image generation was used."
	case "reasoning", "compaction":
		return ""
	default:
		return ""
	}
}

func summarizeCompactedToolExchange(items []map[string]any, index int) (summary string, nextIndex int, ok bool) {
	item := items[index]
	itemType := normalizeResponsesItemType(item)
	if itemType != "function_call" && itemType != "custom_tool_call" {
		return "", index, false
	}
	if index+1 >= len(items) {
		return "", index, false
	}

	next := items[index+1]
	nextType := normalizeResponsesItemType(next)
	if nextType != "function_call_output" && nextType != "custom_tool_call_output" {
		return "", index, false
	}

	callID, _ := item["call_id"].(string)
	nextCallID, _ := next["call_id"].(string)
	if callID == "" || callID != nextCallID {
		return "", index, false
	}

	name, _ := item["name"].(string)
	args := compactSnippet(stringValue(item["arguments"]))
	if args == "" {
		args = compactSnippet(stringValue(item["input"]))
	}
	output := compactSnippet(stringValue(next["output"]))
	switch {
	case args != "" && output != "":
		return fmt.Sprintf("Tool %s(%s) -> %s", name, args, output), index + 1, true
	case output != "":
		return fmt.Sprintf("Tool %s -> %s", name, output), index + 1, true
	case args != "":
		return fmt.Sprintf("Tool %s(%s)", name, args), index + 1, true
	default:
		return fmt.Sprintf("Tool %s executed", name), index + 1, true
	}
}

func extractStructuredCompactedToolExchange(items []map[string]any, index int) (structured []map[string]any, nextIndex int, ok bool) {
	item := items[index]
	itemType := normalizeResponsesItemType(item)
	if itemType != "function_call" && itemType != "custom_tool_call" {
		return nil, index, false
	}
	if index+1 >= len(items) {
		return nil, index, false
	}

	next := items[index+1]
	nextType := normalizeResponsesItemType(next)
	if nextType != "function_call_output" && nextType != "custom_tool_call_output" {
		return nil, index, false
	}

	callID, _ := item["call_id"].(string)
	nextCallID, _ := next["call_id"].(string)
	if callID == "" || callID != nextCallID {
		return nil, index, false
	}

	return []map[string]any{item, next}, index + 1, true
}

type structuredCompactionCandidate struct {
	structured []map[string]any
	headStart  int
	headEnd    int
	tailDrop   int
	score      int
	recency    int
}

func scoreStructuredCompactionCandidate(structured []map[string]any, bonuses ...int) int {
	score := len(structured) * 10
	for _, item := range structured {
		switch normalizeResponsesItemType(item) {
		case "function_call", "custom_tool_call":
			score += 6
		case "function_call_output", "custom_tool_call_output":
			score += 5
		case "message":
			role, _ := item["role"].(string)
			switch role {
			case "user":
				score += 7
			case "assistant":
				score += 6
			}
		}
	}
	for _, bonus := range bonuses {
		score += bonus
	}
	return score
}

func selectStructuredCompactionCandidate(compactedHead, preservedTail []map[string]any, model string) (structuredCompactionCandidate, bool) {
	best := structuredCompactionCandidate{}
	found := false
	consider := func(candidate structuredCompactionCandidate, ok bool) {
		if !ok {
			return
		}
		if !found || candidate.score > best.score || (candidate.score == best.score && candidate.recency > best.recency) {
			best = candidate
			found = true
		}
	}

	if candidate, ok := candidateBoundaryUserToolRun(compactedHead, preservedTail, model); ok {
		consider(candidate, true)
	}
	if candidate, ok := candidateBoundaryUserToolAssistantRun(compactedHead, preservedTail, model); ok {
		consider(candidate, true)
	}
	if candidate, ok := candidateBoundaryMessagePair(compactedHead, preservedTail, model); ok {
		consider(candidate, true)
	}
	for i := 0; i < len(compactedHead); i++ {
		if candidate, ok := candidateStructuredUserToolRun(compactedHead, i, model); ok {
			consider(candidate, true)
		}
		if candidate, ok := candidateStructuredToolExchangeWithAssistant(compactedHead, i, model); ok {
			consider(candidate, true)
		}
		if candidate, ok := candidateStructuredToolExchange(compactedHead, i); ok {
			consider(candidate, true)
		}
		if candidate, ok := candidateStructuredMessagePair(compactedHead, i, model); ok {
			consider(candidate, true)
		}
	}

	return best, found
}

func candidateStructuredToolExchange(items []map[string]any, index int) (structuredCompactionCandidate, bool) {
	structured, nextIndex, ok := extractStructuredCompactedToolExchange(items, index)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: structured,
		headStart:  index,
		headEnd:    nextIndex,
		score:      scoreStructuredCompactionCandidate(structured),
		recency:    index,
	}, true
}

func extractStructuredCompactedToolExchangeWithAssistant(items []map[string]any, index int, model string) (structured []map[string]any, nextIndex int, ok bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return nil, index, false
	}
	pair, nextIndex, ok := extractStructuredCompactedToolExchange(items, index)
	if !ok || nextIndex+1 >= len(items) {
		return nil, index, false
	}

	next := items[nextIndex+1]
	if normalizeResponsesItemType(next) != "message" {
		return nil, index, false
	}
	role, _ := next["role"].(string)
	if role != "assistant" {
		return nil, index, false
	}
	if extractResponsesItemText(next["content"]) == "" {
		return nil, index, false
	}

	return append(pair, next), nextIndex + 1, true
}

func candidateStructuredToolExchangeWithAssistant(items []map[string]any, index int, model string) (structuredCompactionCandidate, bool) {
	structured, nextIndex, ok := extractStructuredCompactedToolExchangeWithAssistant(items, index, model)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: structured,
		headStart:  index,
		headEnd:    nextIndex,
		score:      scoreStructuredCompactionCandidate(structured, 3),
		recency:    index,
	}, true
}

func extractStructuredCompactedUserToolRun(items []map[string]any, index int, model string) (structured []map[string]any, nextIndex int, ok bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return nil, index, false
	}
	if index+1 >= len(items) {
		return nil, index, false
	}

	item := items[index]
	if normalizeResponsesItemType(item) != "message" {
		return nil, index, false
	}
	role, _ := item["role"].(string)
	if role != "user" || extractResponsesItemText(item["content"]) == "" {
		return nil, index, false
	}

	triple, nextIndex, ok := extractStructuredCompactedToolExchangeWithAssistant(items, index+1, model)
	if !ok {
		return nil, index, false
	}

	return append([]map[string]any{item}, triple...), nextIndex, true
}

func candidateStructuredUserToolRun(items []map[string]any, index int, model string) (structuredCompactionCandidate, bool) {
	structured, nextIndex, ok := extractStructuredCompactedUserToolRun(items, index, model)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: structured,
		headStart:  index,
		headEnd:    nextIndex,
		score:      scoreStructuredCompactionCandidate(structured, 8),
		recency:    index,
	}, true
}

func candidateStructuredMessagePair(items []map[string]any, index int, model string) (structuredCompactionCandidate, bool) {
	structured, nextIndex, ok := extractStructuredCompactedMessagePair(items, index, model)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: structured,
		headStart:  index,
		headEnd:    nextIndex,
		score:      scoreStructuredCompactionCandidate(structured),
		recency:    index,
	}, true
}

func extractStructuredCompactedMessagePair(items []map[string]any, index int, model string) (structured []map[string]any, nextIndex int, ok bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return nil, index, false
	}
	if index+1 >= len(items) {
		return nil, index, false
	}

	item := items[index]
	next := items[index+1]
	if normalizeResponsesItemType(item) != "message" || normalizeResponsesItemType(next) != "message" {
		return nil, index, false
	}

	role, _ := item["role"].(string)
	nextRole, _ := next["role"].(string)
	if role != "user" || nextRole != "assistant" {
		return nil, index, false
	}
	if extractResponsesItemText(item["content"]) == "" || extractResponsesItemText(next["content"]) == "" {
		return nil, index, false
	}

	return []map[string]any{item, next}, index + 1, true
}

func candidateBoundaryMessagePair(compactedHead, preservedTail []map[string]any, model string) (structuredCompactionCandidate, bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return structuredCompactionCandidate{}, false
	}
	if len(compactedHead) == 0 || len(preservedTail) == 0 {
		return structuredCompactionCandidate{}, false
	}

	lastHead := compactedHead[len(compactedHead)-1]
	firstTail := preservedTail[0]
	if normalizeResponsesItemType(lastHead) != "message" || normalizeResponsesItemType(firstTail) != "message" {
		return structuredCompactionCandidate{}, false
	}
	role, _ := lastHead["role"].(string)
	nextRole, _ := firstTail["role"].(string)
	if role != "user" || nextRole != "assistant" {
		return structuredCompactionCandidate{}, false
	}
	if extractResponsesItemText(lastHead["content"]) == "" || extractResponsesItemText(firstTail["content"]) == "" {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: []map[string]any{lastHead, firstTail},
		headStart:  len(compactedHead) - 1,
		headEnd:    len(compactedHead) - 1,
		tailDrop:   1,
		score:      scoreStructuredCompactionCandidate([]map[string]any{lastHead, firstTail}, 1),
		recency:    len(compactedHead) - 1,
	}, true
}

func candidateBoundaryUserToolRun(compactedHead, preservedTail []map[string]any, model string) (structuredCompactionCandidate, bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return structuredCompactionCandidate{}, false
	}
	if len(compactedHead) == 0 || len(preservedTail) < 3 {
		return structuredCompactionCandidate{}, false
	}
	lastHead := compactedHead[len(compactedHead)-1]
	if normalizeResponsesItemType(lastHead) != "message" {
		return structuredCompactionCandidate{}, false
	}
	role, _ := lastHead["role"].(string)
	if role != "user" || extractResponsesItemText(lastHead["content"]) == "" {
		return structuredCompactionCandidate{}, false
	}
	triple, nextIndex, ok := extractStructuredCompactedToolExchangeWithAssistant(preservedTail, 0, model)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: append([]map[string]any{lastHead}, triple...),
		headStart:  len(compactedHead) - 1,
		headEnd:    len(compactedHead) - 1,
		tailDrop:   nextIndex + 1,
		score:      scoreStructuredCompactionCandidate(append([]map[string]any{lastHead}, triple...), 6),
		recency:    len(compactedHead) - 1,
	}, true
}

func candidateBoundaryUserToolAssistantRun(compactedHead, preservedTail []map[string]any, model string) (structuredCompactionCandidate, bool) {
	if !allowsStructuredCompactedMessagePair(model) {
		return structuredCompactionCandidate{}, false
	}
	if len(compactedHead) < 3 || len(preservedTail) == 0 {
		return structuredCompactionCandidate{}, false
	}
	user := compactedHead[len(compactedHead)-3]
	firstTail := preservedTail[0]
	if normalizeResponsesItemType(user) != "message" || normalizeResponsesItemType(firstTail) != "message" {
		return structuredCompactionCandidate{}, false
	}
	role, _ := user["role"].(string)
	tailRole, _ := firstTail["role"].(string)
	if role != "user" || tailRole != "assistant" {
		return structuredCompactionCandidate{}, false
	}
	pair, _, ok := extractStructuredCompactedToolExchange(compactedHead, len(compactedHead)-2)
	if !ok {
		return structuredCompactionCandidate{}, false
	}
	if extractResponsesItemText(user["content"]) == "" || extractResponsesItemText(firstTail["content"]) == "" {
		return structuredCompactionCandidate{}, false
	}
	return structuredCompactionCandidate{
		structured: append([]map[string]any{user}, append(pair, firstTail)...),
		headStart:  len(compactedHead) - 3,
		headEnd:    len(compactedHead) - 1,
		tailDrop:   1,
		score:      scoreStructuredCompactionCandidate(append([]map[string]any{user}, append(pair, firstTail)...), 7),
		recency:    len(compactedHead) - 1,
	}, true
}

func allowsStructuredCompactedMessagePair(model string) bool {
	limit, ok := lookupCloudModelLimit(model)
	return ok && limit.Context >= 200_000
}

func summarizeCompactedToolExchangeWithAssistant(items []map[string]any, index int) (summary string, nextIndex int, ok bool) {
	toolSummary, nextIndex, ok := summarizeCompactedToolExchange(items, index)
	if !ok {
		return "", index, false
	}
	if nextIndex+1 >= len(items) {
		return "", index, false
	}

	next := items[nextIndex+1]
	if normalizeResponsesItemType(next) != "message" {
		return "", index, false
	}
	role, _ := next["role"].(string)
	if role != "assistant" {
		return "", index, false
	}

	text := extractResponsesItemText(next["content"])
	if text == "" {
		return "", index, false
	}

	return fmt.Sprintf("%s | Assistant: %s", toolSummary, text), nextIndex + 1, true
}

func summarizeCompactedUserToolAssistantRun(items []map[string]any, index int) (summary string, nextIndex int, ok bool) {
	item := items[index]
	if normalizeResponsesItemType(item) != "message" {
		return "", index, false
	}
	role, _ := item["role"].(string)
	if role != "user" {
		return "", index, false
	}

	userText := extractResponsesItemText(item["content"])
	if userText == "" || index+1 >= len(items) {
		return "", index, false
	}

	toolSummary, toolNextIndex, ok := summarizeCompactedToolExchangeWithAssistant(items, index+1)
	if !ok {
		return "", index, false
	}

	return fmt.Sprintf("User: %s | %s", userText, toolSummary), toolNextIndex, true
}

func summarizeCompactedMessageRun(items []map[string]any, index int) (summary string, nextIndex int, ok bool) {
	item := items[index]
	if normalizeResponsesItemType(item) != "message" {
		return "", index, false
	}

	role, _ := item["role"].(string)
	if role != "user" && role != "assistant" {
		return "", index, false
	}

	lines := make([]string, 0, 3)
	currentRole := role
	currentText := extractResponsesItemText(item["content"])
	if currentText == "" {
		return "", index, false
	}
	lines = append(lines, fmt.Sprintf("%s: %s", roleLabel(currentRole), currentText))
	lastText := currentText
	last := index

	for j := index + 1; j < len(items) && len(lines) < 3; j++ {
		next := items[j]
		if normalizeResponsesItemType(next) != "message" {
			break
		}
		nextRole, _ := next["role"].(string)
		if nextRole != "user" && nextRole != "assistant" {
			break
		}
		if nextRole == currentRole {
			break
		}
		nextText := extractResponsesItemText(next["content"])
		if nextText == "" {
			break
		}
		if nextText == lastText {
			// Skip duplicate text but do NOT advance currentRole; advancing it
			// would allow the next message to bypass the same-role check and
			// produce a confusing same-role pair in the summary output.
			last = j
			continue
		}
		lines = append(lines, fmt.Sprintf("%s: %s", roleLabel(nextRole), nextText))
		currentRole = nextRole
		lastText = nextText
		last = j
	}

	if len(lines) < 2 {
		return "", index, false
	}

	return strings.Join(lines, " | "), last, true
}

func roleLabel(role string) string {
	switch role {
	case "assistant":
		return "Asst"
	case "user":
		return "User"
	default:
		return role
	}
}

func summaryDuplicatesPreserved(summary string, preservedTail []map[string]any) bool {
	for _, item := range preservedTail {
		text := extractResponsesItemText(item["content"])
		// Require a minimum length to avoid false positives where a short
		// common word (e.g. "ok", "yes") in a tail item incorrectly matches
		// an unrelated summary part that merely contains that substring.
		if len(text) >= 20 && strings.Contains(summary, text) {
			return true
		}
	}
	return false
}

func summarizeCompactedSameRoleMessageRun(items []map[string]any, index int) (summary string, nextIndex int, ok bool) {
	item := items[index]
	if normalizeResponsesItemType(item) != "message" {
		return "", index, false
	}

	role, _ := item["role"].(string)
	if role != "user" && role != "assistant" {
		return "", index, false
	}

	lines := make([]string, 0, responsesCompactSameRoleRunMaxItems)
	text := extractResponsesItemText(item["content"])
	if text == "" {
		return "", index, false
	}
	lines = append(lines, text)
	last := index

	for j := index + 1; j < len(items) && len(lines) < responsesCompactSameRoleRunMaxItems; j++ {
		next := items[j]
		if normalizeResponsesItemType(next) != "message" {
			break
		}
		nextRole, _ := next["role"].(string)
		if nextRole != role {
			break
		}
		nextText := extractResponsesItemText(next["content"])
		if nextText == "" {
			break
		}
		if nextText == lines[len(lines)-1] {
			last = j
			continue
		}
		if len(strings.Join(append(append([]string{}, lines...), nextText), " / ")) > responsesCompactSameRoleRunMaxChars {
			break
		}
		lines = append(lines, nextText)
		last = j
	}

	if len(lines) < 2 {
		return "", index, false
	}

	return fmt.Sprintf("%s: %s", roleLabel(role), strings.Join(lines, " / ")), last, true
}

func buildCompactionSummary(parts []string, omittedCounts map[string]int) string {
	return buildCompactionSummaryWithTurnCount(parts, omittedCounts, 0)
}

func buildCompactionSummaryWithTurnCount(parts []string, omittedCounts map[string]int, turnCount int) string {
	if len(parts) == 0 && len(omittedCounts) == 0 {
		return ""
	}
	var compactedHistoryPrefix string
	if turnCount > 0 {
		compactedHistoryPrefix = fmt.Sprintf("Previous conversation history was compacted by Ollama (summarising %d turns).\n\n", turnCount)
	} else {
		compactedHistoryPrefix = "Previous conversation history was compacted by Ollama.\n\n"
	}
	const preservedSummaryHeader = "Preserved summary:\n"

	var b strings.Builder
	b.WriteString(compactedHistoryPrefix)

	additionalItemsSection := buildCompactionOmittedCountsSection(omittedCounts)
	summaryBudget := responsesCompactSummaryMaxChars - len(additionalItemsSection)
	if additionalItemsSection != "" {
		summaryBudget--
	}
	if summaryBudget < len(compactedHistoryPrefix) {
		summaryBudget = len(compactedHistoryPrefix)
	}

	if len(parts) > 0 {
		selected := selectCompactionSummaryParts(parts, summaryBudget-len(compactedHistoryPrefix)-len(preservedSummaryHeader))
		if len(selected) > 0 {
			b.WriteString(preservedSummaryHeader)
			for _, part := range selected {
				b.WriteString("- ")
				b.WriteString(part)
				b.WriteByte('\n')
			}
		}
	}

	if additionalItemsSection != "" {
		if b.Len() > len(compactedHistoryPrefix) {
			b.WriteByte('\n')
		}
		b.WriteString(additionalItemsSection)
	}

	summary := strings.TrimSpace(b.String())
	if len(summary) <= responsesCompactSummaryMaxChars {
		return summary
	}
	return strings.TrimSpace(summary[:responsesCompactSummaryMaxChars-3]) + "..."
}

// responsesCompactSummaryMaxItems caps the number of bullet points in a
// compaction summary to keep it readable and avoid overwhelming the model.
const responsesCompactSummaryMaxItems = 80

func selectCompactionSummaryParts(parts []string, budget int) []string {
	if budget <= 0 {
		return nil
	}

	type summaryCandidate struct {
		text    string
		index   int
		score   int
		kind    string
		density int
		lineLen int
	}

	candidateByKey := make(map[string]summaryCandidate, len(parts))
	for i, raw := range parts {
		part := compactSnippet(raw)
		if part == "" {
			continue
		}
		key := normalizeCompactionSummaryPart(part)
		candidate := summaryCandidate{
			text:    part,
			index:   i,
			score:   scoreCompactionSummaryPart(part),
			kind:    classifyCompactionSummaryPart(part),
			lineLen: len("- ") + len(part) + 1,
		}
		candidate.density = scoreCompactionSummaryDensity(candidate.score, candidate.lineLen)
		if existing, ok := candidateByKey[key]; ok {
			if existing.score > candidate.score || (existing.score == candidate.score && (existing.density > candidate.density || (existing.density == candidate.density && existing.index > candidate.index))) {
				continue
			}
		}
		candidateByKey[key] = candidate
	}
	candidates := make([]summaryCandidate, 0, len(candidateByKey))
	for _, candidate := range candidateByKey {
		candidates = append(candidates, candidate)
	}
	slices.SortFunc(candidates, func(a, b summaryCandidate) int {
		if a.score != b.score {
			if a.score > b.score {
				return -1
			}
			return 1
		}
		if a.density != b.density {
			if a.density > b.density {
				return -1
			}
			return 1
		}
		if a.index != b.index {
			if a.index > b.index {
				return -1
			}
			return 1
		}
		return 0
	})

	// Single O(N log N) sorted pass: candidates are already sorted by
	// (score desc, density desc, index desc). We iterate once, applying the
	// kind-diversity penalty on the fly. Because the penalty values are small
	// relative to the score gaps between kinds, this produces results
	// equivalent to the previous O(N²) greedy loop for all practical inputs
	// while being significantly faster for long conversations.
	selected := make([]summaryCandidate, 0, len(candidates))
	used := 0
	selectedKinds := make(map[string]int, 4)
	for _, candidate := range candidates {
		if len(selected) >= responsesCompactSummaryMaxItems {
			break
		}
		if used+candidate.lineLen > budget {
			continue
		}
		effective := candidate.score - selectedKinds[candidate.kind]*compactionSummaryKindPenalty(candidate.kind)
		if effective <= 0 {
			continue
		}
		selected = append(selected, candidate)
		selectedKinds[candidate.kind]++
		used += candidate.lineLen
	}

	slices.SortFunc(selected, func(a, b summaryCandidate) int {
		if a.index < b.index {
			return -1
		}
		if a.index > b.index {
			return 1
		}
		return 0
	})

	output := make([]string, 0, len(selected))
	for _, candidate := range selected {
		output = append(output, candidate.text)
	}
	return output
}

func buildCompactionOmittedCountsSection(omittedCounts map[string]int) string {
	if len(omittedCounts) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("Additional compacted items:")
	keys := make([]string, 0, len(omittedCounts))
	for key := range omittedCounts {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		b.WriteString(fmt.Sprintf(" %s=%d", key, omittedCounts[key]))
	}
	return b.String()
}

func scoreCompactionSummaryPart(part string) int {
	score := 0
	switch {
	case strings.Contains(part, " | Tool ") && strings.Contains(part, " | Assistant:"):
		score += 60
	case strings.Contains(part, "Tool ") && strings.Contains(part, "->"):
		score += 45
	case strings.Contains(part, "User:") && strings.Contains(part, "Assistant:"):
		score += 35
	case strings.Contains(part, "Assistant:"):
		score += 20
	case strings.Contains(part, "User:"):
		score += 18
	case strings.Contains(part, "Tool call ") || strings.Contains(part, "Tool output:"):
		score += 15
	}
	score += min(len(part)/24, 10)
	return score
}

func classifyCompactionSummaryPart(part string) string {
	switch {
	case strings.Contains(part, " | Tool ") && strings.Contains(part, " | Assistant:"):
		return "user_tool_assistant"
	case strings.Contains(part, "Tool ") && strings.Contains(part, "->"):
		return "tool_exchange"
	case strings.Contains(part, "User:") && strings.Contains(part, "Assistant:"):
		return "message_pair"
	case strings.Contains(part, "Assistant:"):
		return "assistant"
	case strings.Contains(part, "User:"):
		return "user"
	case strings.Contains(part, "Tool call ") || strings.Contains(part, "Tool output:"):
		return "tool"
	default:
		return "other"
	}
}

func compactionSummaryKindPenalty(kind string) int {
	switch kind {
	case "user_tool_assistant":
		return 6
	case "tool_exchange", "message_pair":
		return 8
	case "assistant", "user":
		return 12
	case "tool":
		return 14
	default:
		return 10
	}
}

func scoreCompactionSummaryDensity(score, lineLen int) int {
	if lineLen <= 0 {
		return score * 100
	}
	return score * 100 / lineLen
}

func normalizeCompactionSummaryPart(part string) string {
	part = strings.TrimSpace(part)
	prefixes := []string{
		"Assistant:",
		"User:",
		"Tool output:",
		"Tool call:",
		"Web search:",
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(part, prefix) {
			part = strings.TrimSpace(strings.TrimPrefix(part, prefix))
			goto trimmed
		}
	}
	if strings.HasPrefix(part, "Tool ") {
		part = strings.TrimSpace(strings.TrimPrefix(part, "Tool "))
	}
trimmed:
	leadIns := []string{
		"confirmed ",
		"confirming ",
		"noted ",
		"noting ",
		"result: ",
		"summary: ",
	}
	lower := strings.ToLower(part)
	for _, leadIn := range leadIns {
		if strings.HasPrefix(lower, leadIn) {
			part = strings.TrimSpace(part[len(leadIn):])
			break
		}
	}
	part = strings.NewReplacer(
		"“", `"`,
		"”", `"`,
		"‘", `'`,
		"’", `'`,
	).Replace(part)
	part = strings.Join(strings.Fields(part), " ")
	part = stripOuterBrackets(part)
	part = stripOuterQuotes(part)
	part = strings.TrimSpace(strings.TrimRight(part, " .,!?:;"))
	return part
}

func compactSnippet(text string) string {
	text = strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
	if len(text) <= 240 {
		return text
	}
	return text[:237] + "..."
}

func extractResponsesItemText(content any) string {
	raw := extractResponsesItemRawText(content)
	if raw == "" {
		return ""
	}
	return compactSnippet(raw)
}

func extractResponsesItemRawText(content any) string {
	switch typed := content.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []any:
		parts := make([]string, 0, len(typed))
		for _, raw := range typed {
			item, _ := raw.(map[string]any)
			if item == nil {
				continue
			}
			text := strings.TrimSpace(stringValue(item["text"]))
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func makeResponsesInputMessage(role, text string) map[string]any {
	return map[string]any{
		"type": "message",
		"role": role,
		"content": []map[string]any{
			{
				"type": "input_text",
				"text": text,
			},
		},
	}
}

func makeResponsesAssistantMessage(text string) map[string]any {
	return map[string]any{
		"type": "message",
		"role": "assistant",
		"content": []map[string]any{
			{
				"type": "output_text",
				"text": text,
			},
		},
	}
}

// stripOuterBrackets delegates to internal/strutil.StripOuterBrackets.
// The implementation lives there so it can be tested independently of the
// server package's CGO (MLX) build constraints.
func stripOuterBrackets(s string) string { return strutil.StripOuterBrackets(s) }

// stripOuterQuotes delegates to internal/strutil.StripOuterQuotes.
func stripOuterQuotes(s string) string { return strutil.StripOuterQuotes(s) }
