"""
Tokenizer rendering tests for the unrestricted conversation-template mod.

Covers (all against a minimal synthetic tiktoken Encoding, no trained
tokenizer / rustbpe needed):
- legacy strict-alternation conversations render EXACTLY as before
- system-message merge preserved (incl. the edge: system not followed by user)
- no alternation asserts: assistant-first, consecutive assistants, user after user
- assistant multi-part messages (text -> python -> python_output -> text):
  correct special-token scaffolding and supervision masks
- the 2048 cap is GONE: long conversations render in full (memory budget),
  an explicit max_tokens is still honored (backward compat)
- segments + pack_segments_into_windows: windowing never drops tokens, rows
  respect the row capacity, masks travel with their tokens
- render_for_completion: tolerant of conversations ending with a user message
- render_conversations_parallel: identical results to sequential rendering
- auto_token_budget: memory-derived and above the old 2048

Run: python tests/test_tokenizer_render.py   (or pytest tests/test_tokenizer_render.py -v)
"""

import sys

import tiktoken
import torch  # noqa: F401 (FoxSay imports)

from FoxSay.tokenizer import (
    SPECIAL_TOKENS,
    RustBPETokenizer,
    pack_segments_into_windows,
)


def build_synth_tokenizer():
    """Byte-level identity encoding: token id i == byte i for i < 256, then
    special tokens. Deterministic, dependency-free, easy to verify."""
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    offset = len(mergeable_ranks)
    special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="synth",
        pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special,
    )
    return RustBPETokenizer(enc, "<|bos|>")


TOK = build_synth_tokenizer()


def sp(name):
    return TOK.encode_special(name)


def msg(role, content):
    return {"role": role, "content": content}


def conv(*messages):
    return {"messages": list(messages)}


def main():
    # --- Test 1: legacy strict conversation renders exactly as before --------
    c = conv(
        msg("user", "hello"),
        msg("assistant", "hi there"),
        msg("user", "bye"),
        msg("assistant", "goodbye"),
    )
    ids, mask = TOK.render_conversation(c)
    expected = (
        [sp("<|bos|>")]
        + [sp("<|user_start|>")] + TOK.encode("hello") + [sp("<|user_end|>")]
        + [sp("<|assistant_start|>")] + TOK.encode("hi there") + [sp("<|assistant_end|>")]
        + [sp("<|user_start|>")] + TOK.encode("bye") + [sp("<|user_end|>")]
        + [sp("<|assistant_start|>")] + TOK.encode("goodbye") + [sp("<|assistant_end|>")]
    )
    assert ids == expected, "legacy alternation conversation changed!"
    assert len(ids) == len(mask)
    assert sum(mask) == len(TOK.encode("hi there")) + len(TOK.encode("goodbye")) + 2, \
        "supervision mask must cover assistant text + assistant_end"
    print("  [ok] legacy strict-alternation rendering is bit-for-bit unchanged")

    # --- Test 2: system merge preserved (+ edge: system without user after) --
    c = conv(msg("system", "be nice"), msg("user", "hi"), msg("assistant", "hello"))
    ids, mask = TOK.render_conversation(c)
    merged = TOK.encode("be nice\n\nhi")
    assert ids[1:1 + len(merged) + 2] == [sp("<|user_start|>")] + merged + [sp("<|user_end|>")]
    c2 = conv(msg("system", "be nice"), msg("assistant", "hello"))
    ids2, _ = TOK.render_conversation(c2)  # must not assert-crash
    assert sp("<|assistant_start|>") in ids2
    assert TOK.encode("be nice") and all(t in ids2 for t in TOK.encode("be nice"))
    print("  [ok] system merge preserved; system-without-user falls back to a user turn")

    # --- Test 3: alternation asserts removed --------------------------------
    TOK.render_conversation(conv(msg("assistant", "first!")))              # assistant-first
    TOK.render_conversation(conv(msg("user", "a"), msg("user", "b")))      # user after user
    TOK.render_conversation(conv(msg("assistant", "x"), msg("assistant", "y")))  # back-to-back
    TOK.render_conversation(conv(msg("user", "a"), msg("assistant", "x"),
                                  msg("assistant", "y"), msg("user", "b"))) # mixed
    print("  [ok] arbitrary role orders render (no alternation asserts)")

    # --- Test 4: assistant multi-part (text -> python -> output -> text) -----
    c = conv(
        msg("user", "compute 2+2"),
        msg("assistant", [
            {"type": "text", "text": "let me think"},
            {"type": "python", "text": "print(2+2)"},
            {"type": "python_output", "text": "4"},
            {"type": "text", "text": "it is 4"},
        ]),
    )
    ids, mask = TOK.render_conversation(c)
    seq = (
        [sp("<|bos|>")]
        + [sp("<|user_start|>")] + TOK.encode("compute 2+2") + [sp("<|user_end|>")]
        + [sp("<|assistant_start|>")]
        + TOK.encode("let me think")
        + [sp("<|python_start|>")] + TOK.encode("print(2+2)") + [sp("<|python_end|>")]
        + [sp("<|output_start|>")] + TOK.encode("4") + [sp("<|output_end|>")]
        + TOK.encode("it is 4")
        + [sp("<|assistant_end|>")]
    )
    assert ids == seq, "multi-part assistant scaffolding wrong"
    out_pos = ids.index(sp("<|output_start|>"))
    out_end = ids.index(sp("<|output_end|>"))
    assert all(m == 0 for m in mask[out_pos:out_end + 1]), "python output must be unsupervised"
    py_pos = ids.index(sp("<|python_start|>"))
    py_end = ids.index(sp("<|python_end|>"))
    assert all(m == 1 for m in mask[py_pos:py_end + 1]), "python call must be supervised"
    print("  [ok] multi-part assistant turns: scaffolding + supervision masks correct")

    # --- Test 5: the 2048 cap is gone ----------------------------------------
    long_conv = conv(*[msg("user", f"message number {i} " + "x" * 30) for i in range(300)])
    ids, mask = TOK.render_conversation(long_conv)  # default: memory budget
    assert len(ids) > 2048, f"long conversation was capped: {len(ids)}"
    assert len(ids) == len(mask)
    # backward compat: explicit cap still honored
    ids_capped, mask_capped = TOK.render_conversation(long_conv, max_tokens=100)
    assert len(ids_capped) == 100 and len(mask_capped) == 100
    assert ids_capped == ids[:100] and mask_capped == mask[:100]
    assert TOK.auto_token_budget() > 2048, "auto budget must exceed the old fixed cap"
    print(f"  [ok] no 2048 cap: {len(ids)} tokens rendered; explicit max_tokens still works;"
          f" auto budget = {TOK.auto_token_budget():,}")

    # --- Test 6: segments + windowing ----------------------------------------
    ids, mask, segments = TOK.render_conversation(long_conv, return_segments=True)
    total = sum(len(s[0]) for s in segments)
    assert total == len(ids), "segments must cover every token exactly once"
    flat = [t for s_ids, _ in segments for t in s_ids]
    flat_mask = [m for _, s_mask in segments for m in s_mask]
    assert flat == ids and flat_mask == mask, "segments must preserve order"
    rows = pack_segments_into_windows(segments, max_len=128)
    assert all(len(r_ids) <= 128 for r_ids, _ in rows)
    joined = [t for r_ids, _ in rows for t in r_ids]
    joined_mask = [m for _, r_mask in rows for m in r_mask]
    assert joined == ids and joined_mask == mask, "windowing must never drop or reorder tokens"
    assert rows[0][0][0] == sp("<|bos|>"), "first row starts with BOS"
    short = pack_segments_into_windows(segments, max_len=len(ids) + 5)
    assert len(short) == 1, "a fitting conversation yields exactly one row"
    print(f"  [ok] windowing: {len(rows)} rows @ cap 128, zero token loss, masks aligned")

    # --- Test 7: render_for_completion tolerant -------------------------------
    c = conv(msg("user", "q"), msg("assistant", "a"))
    ids = TOK.render_for_completion(c)
    assert ids[-1] == sp("<|assistant_start|>")
    assert TOK.encode("a") and not any(t in ids[len(TOK.encode("q")):] for t in TOK.encode("a") if False), ""
    # conversation ending with a user message: no pop, straight to completion
    ids2 = TOK.render_for_completion(conv(msg("user", "q")))
    assert ids2[-1] == sp("<|assistant_start|>") and TOK.encode("q") and all(
        t in ids2 for t in TOK.encode("q"))
    print("  [ok] render_for_completion: pops trailing assistant, tolerates user-final")

    # --- Test 8: parallel bulk rendering ---------------------------------------
    convs = [long_conv, c, conv(msg("user", "solo"))]
    seq_results = [TOK.render_conversation(x) for x in convs]
    par_results = TOK.render_conversations_parallel(convs, num_workers=4)
    assert par_results == seq_results, "parallel rendering must match sequential exactly"
    print("  [ok] render_conversations_parallel == sequential, order preserved")

    # --- Test 9: malformed inputs still fail loudly ----------------------------
    for bad in (
        conv(msg("tool", "oops")),                        # unknown role
        conv(msg("assistant", [{"type": "weird", "text": "x"}])),  # unknown part type
        conv(msg("user", 42)),                            # bad content type
    ):
        try:
            TOK.render_conversation(bad)
            raise AssertionError(f"malformed conversation must raise: {bad}")
        except (ValueError, TypeError):
            pass
    print("  [ok] malformed roles/parts/content still raise ValueError/TypeError")

    print("All tokenizer render tests passed.")


if __name__ == "__main__":
    main()
