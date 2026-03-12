package server

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	responsesCompactKeepRecentUserMessages  = 1
	responsesCompactKeepRecentAssistantMsgs = 1
	responsesCompactSummaryMaxChars         = 1200
)

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
		"id":         fmt.Sprintf("respcomp_%d", rand.Intn(999999)),
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
	userMessages := make([]map[string]any, 0, len(items))
	assistantMessages := make([]map[string]any, 0, len(items))
	summaryParts := make([]string, 0, len(items))
	omittedCounts := map[string]int{}

	for _, item := range items {
		itemType := normalizeResponsesItemType(item)
		role, _ := item["role"].(string)

		if itemType == "message" && slices.Contains([]string{"system", "developer"}, role) {
			systemAndDeveloper = append(systemAndDeveloper, item)
			continue
		}

		if itemType == "message" && role == "user" {
			userMessages = append(userMessages, item)
			continue
		}

		if itemType == "message" && role == "assistant" {
			assistantMessages = append(assistantMessages, item)
			continue
		}

		summary := summarizeCompactedInputItem(itemType, item)
		if summary != "" {
			summaryParts = append(summaryParts, summary)
		} else {
			omittedCounts[itemType]++
		}
	}

	output := make([]map[string]any, 0, len(systemAndDeveloper)+len(userMessages)+len(assistantMessages)+1)
	output = append(output, systemAndDeveloper...)

	if len(userMessages) > responsesCompactKeepRecentUserMessages {
		droppedUsers := userMessages[:len(userMessages)-responsesCompactKeepRecentUserMessages]
		for _, item := range droppedUsers {
			if summary := summarizeCompactedInputItem("message", item); summary != "" {
				summaryParts = append(summaryParts, summary)
			} else {
				omittedCounts["user_message"]++
			}
		}
		userMessages = userMessages[len(userMessages)-responsesCompactKeepRecentUserMessages:]
	}
	output = append(output, userMessages...)

	if len(assistantMessages) > responsesCompactKeepRecentAssistantMsgs {
		droppedAssistants := assistantMessages[:len(assistantMessages)-responsesCompactKeepRecentAssistantMsgs]
		for _, item := range droppedAssistants {
			if summary := summarizeCompactedInputItem("message", item); summary != "" {
				summaryParts = append(summaryParts, summary)
			} else {
				omittedCounts["assistant_message"]++
			}
		}
		assistantMessages = assistantMessages[len(assistantMessages)-responsesCompactKeepRecentAssistantMsgs:]
	}
	output = append(output, assistantMessages...)

	if summaryText := buildCompactionSummary(summaryParts, omittedCounts); summaryText != "" {
		output = append(output, makeResponsesAssistantMessage(summaryText))
	}

	if len(output) == 0 {
		output = append(output, makeResponsesAssistantMessage("Previous conversation history was compacted by Ollama."))
	}

	return output, nil
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

func buildCompactionSummary(parts []string, omittedCounts map[string]int) string {
	if len(parts) == 0 && len(omittedCounts) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("Previous conversation history was compacted by Ollama.\n\n")

	if len(parts) > 0 {
		b.WriteString("Preserved summary:\n")
		for _, part := range parts {
			if b.Len() >= responsesCompactSummaryMaxChars {
				break
			}
			b.WriteString("- ")
			b.WriteString(compactSnippet(part))
			b.WriteByte('\n')
		}
	}

	if len(omittedCounts) > 0 {
		if len(parts) > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("Additional compacted items:")
		keys := make([]string, 0, len(omittedCounts))
		for key := range omittedCounts {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		for _, key := range keys {
			b.WriteString(fmt.Sprintf(" %s=%d", key, omittedCounts[key]))
		}
	}

	summary := strings.TrimSpace(b.String())
	if len(summary) <= responsesCompactSummaryMaxChars {
		return summary
	}
	return strings.TrimSpace(summary[:responsesCompactSummaryMaxChars-3]) + "..."
}

func compactSnippet(text string) string {
	text = strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
	if len(text) <= 240 {
		return text
	}
	return text[:237] + "..."
}

func extractResponsesItemText(content any) string {
	switch typed := content.(type) {
	case string:
		return compactSnippet(typed)
	case []any:
		parts := make([]string, 0, len(typed))
		for _, raw := range typed {
			item, _ := raw.(map[string]any)
			if item == nil {
				continue
			}
			text := compactSnippet(stringValue(item["text"]))
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
