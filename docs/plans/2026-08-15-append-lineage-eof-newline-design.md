# Append Recovery EOF Newline Lineage Design

## Problem

`append_recovery` preserves the previous program as a prefix and inserts a line boundary before
the recovery suffix. When the previous source has no trailing newline, segmentation changes the
stored source text of the final pre-append region and group from `statement()` to `statement()\n`.
Lineage reconciliation currently requires exact source-string equality, so an already-executed
final unit cannot be mapped even though its executable content and line span are unchanged.

## Approved Scope

Keep exact source matching everywhere except at the old EOF during `append_recovery`. For the
final old region or group (`end_line == old_line_count`), treat sources as equal when they differ
only by trailing CR/LF characters. Start and end lines must still match exactly. All non-append
edits, non-final units, and substantive source changes retain the existing strict behavior.

## Implementation

Add a small lineage source-matching helper used by `_reconcile_units`. It first accepts exact
equality. Its only fallback applies when `edit_kind == "append_recovery"` and the previous unit
ends at the old source boundary; the fallback compares the two unit sources after removing only
terminal `\r` and `\n` characters.

Apply the rule to both regions and groups. The experiment exposed the region failure first, but
the final executed group receives the same newline from segmentation and would otherwise fail on
the next reconciliation pass.

## Safety and Tests

- Reproduce the observed failure with an executed final region and group whose candidate sources
  gain one terminal newline during append.
- Assert that both stable keys remain mapped and executed.
- Assert that changing any executable text in the final unit is still rejected.
- Run the focused lineage tests and the runtime-control trial-loop regression suite in WSL.
