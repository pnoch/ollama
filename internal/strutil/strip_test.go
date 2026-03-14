package strutil_test

import (
	"testing"

	"github.com/ollama/ollama/internal/strutil"
)

func TestStripOuterBrackets(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"(hello)", "hello"},
		{"[world]", "world"},
		{"{foo}", "foo"},
		{"((nested) text)", "(nested) text"},
		{"(hello) world (test)", "(hello) world (test)"},
		{"(unmatched", "(unmatched"},
		{"unmatched)", "unmatched)"},
		{"", ""},
		{"x", "x"},
		{"(  spaces  )", "spaces"},
		{"((double))", "double"},
		{"[a (b) c]", "a (b) c"},
		{"(a)(b)", "(a)(b)"},
		{"( )", ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := strutil.StripOuterBrackets(tt.input)
			if got != tt.expected {
				t.Errorf("StripOuterBrackets(%q) = %q, want %q", tt.input, got, tt.expected)
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
		{`'world'`, "world"},
		{`"hello" world "test"`, `"hello" world "test"`},
		{`'it's fine'`, `'it's fine'`},
		// When inner quotes are escaped (backslash-prefixed), the outer quotes
		// ARE a matched pair and should be stripped.
		{`"he said \"hi\""`, `he said \"hi\"`},
		{`"unmatched`, `"unmatched`},
		{`unmatched"`, `unmatched"`},
		{"", ""},
		{"x", "x"},
		{`"  spaces  "`, "spaces"},
		{`""`, ""},
		{`"a"`, "a"},
		{`'a'`, "a"},
		{`"a'b"`, "a'b"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := strutil.StripOuterQuotes(tt.input)
			if got != tt.expected {
				t.Errorf("StripOuterQuotes(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestContainsUnescapedByte(t *testing.T) {
	tests := []struct {
		s        string
		b        byte
		expected bool
	}{
		{"hello", 'l', true},
		{"hello", 'z', false},
		{`he said \"hi\"`, '"', false},
		{`he said "hi"`, '"', true},
		{`\\`, '\\', false}, // escaped backslash
		{"", '"', false},
	}
	for _, tt := range tests {
		got := strutil.ContainsUnescapedByte(tt.s, tt.b)
		if got != tt.expected {
			t.Errorf("ContainsUnescapedByte(%q, %q) = %v, want %v", tt.s, tt.b, got, tt.expected)
		}
	}
}
