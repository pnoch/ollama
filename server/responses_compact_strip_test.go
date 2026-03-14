package server

import (
	"testing"
)

func TestStripOuterBrackets(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"(hello)", "hello"},
		{"((hello))", "hello"},
		{"(((hello)))", "hello"},
		{"(hello world)", "hello world"},
		{"[item]", "item"},
		{"{key}", "key"},
		// Must NOT strip: closing bracket belongs to inner expression, not outer
		{"(hello) world (test)", "(hello) world (test)"},
		// Outer wraps inner partial — strip outer, leave inner
		{"((nested) text)", "(nested) text"},
		// Outer wraps two inner groups — strip outer, leave inner
		{"((a) (b))", "(a) (b)"},
		// Outer wraps complete inner group — strip outer, leave inner
		{"(a (b) c)", "a (b) c"},
		// Edge cases
		{"", ""},
		{"()", ""},
		{"(a)", "a"},
		{"no brackets", "no brackets"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := stripOuterBrackets(tt.input)
			if got != tt.expected {
				t.Errorf("stripOuterBrackets(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestStripOuterQuotes(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"hello"`, "hello"},
		{`"hello world"`, "hello world"},
		{"'hello'", "hello"},
		// Must NOT strip: same quote character appears inside
		{`"hello" world "test"`, `"hello" world "test"`},
		// Must NOT strip: apostrophe inside single-quoted string
		{"'it's fine'", "'it's fine'"},
		// Must NOT strip: ambiguous double-double-quoted string has inner quotes
		{`""hello""`, `""hello""`},
		// Must NOT strip: unmatched quotes
		{`"no close`, `"no close`},
		// Edge cases
		{"", ""},
		{`""`, ""},
		{"plain text", "plain text"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := stripOuterQuotes(tt.input)
			if got != tt.expected {
				t.Errorf("stripOuterQuotes(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}
