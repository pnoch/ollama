// Package strutil provides small string utility helpers used across packages.
package strutil

import "strings"

// StripOuterBrackets removes a single layer of matching outer bracket pairs
// from s. It only strips if the closing bracket at the end of the string is
// the genuine pair of the opening bracket at the start (verified by depth
// counting). Whitespace inside the brackets is trimmed after stripping.
//
// Examples:
//
//	StripOuterBrackets("(hello)")           → "hello"
//	StripOuterBrackets("((nested) text)")   → "(nested) text"
//	StripOuterBrackets("(hello) world (x)") → "(hello) world (x)"  // not stripped
func StripOuterBrackets(s string) string {
	for len(s) >= 2 {
		var open, close byte
		switch s[0] {
		case '(':
			open, close = '(', ')'
		case '[':
			open, close = '[', ']'
		case '{':
			open, close = '{', '}'
		default:
			return s
		}
		if s[len(s)-1] != close {
			return s
		}
		// Verify the closing bracket at the end is the pair of the opening
		// bracket at the start using depth counting.
		depth := 0
		paired := false
		for i := 0; i < len(s); i++ {
			if s[i] == open {
				depth++
			} else if s[i] == close {
				depth--
				if depth == 0 {
					paired = i == len(s)-1
					break
				}
			}
		}
		if !paired {
			return s
		}
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	return s
}

// StripOuterQuotes removes a single layer of matching outer quote characters
// from s (either `"` or `'`). It only strips if the same quote character does
// not appear unescaped inside the string, which would indicate the outer
// quotes are not a matched pair around the whole string.
//
// Examples:
//
//	StripOuterQuotes(`"hello"`)              → "hello"
//	StripOuterQuotes(`"hello" world "test"`) → `"hello" world "test"`  // not stripped
//	StripOuterQuotes(`"he said \"hi\""`)     → `"he said \"hi\""`      // not stripped
func StripOuterQuotes(s string) string {
	for len(s) >= 2 {
		var q byte
		switch s[0] {
		case '"':
			q = '"'
		case '\'':
			q = '\''
		default:
			return s
		}
		if s[len(s)-1] != q {
			return s
		}
		inner := s[1 : len(s)-1]
		// Only strip if the same quote character does not appear unescaped
		// inside, which would mean the outer quotes are not a matched pair
		// around the whole string.
		if ContainsUnescapedByte(inner, q) {
			return s
		}
		s = strings.TrimSpace(inner)
	}
	return s
}

// ContainsUnescapedByte reports whether b appears in s without a preceding
// backslash. It is used to guard against stripping quotes around strings that
// contain escaped quote characters (e.g. `"he said \"hello\""`).
func ContainsUnescapedByte(s string, b byte) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' {
			i++ // skip the escaped character
			continue
		}
		if s[i] == b {
			return true
		}
	}
	return false
}
