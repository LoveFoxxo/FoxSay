"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import sys
import pickle
import tiktoken

try:
    import rustbpe  # only needed by the training path (train_from_iterator)
except ImportError:  # allow inference-only environments (tests, edge boxes)
    rustbpe = None


def available_memory_bytes():
    """Best-effort cross-platform *available* physical memory, in bytes."""
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys)
    try:  # linux (and most POSIX): pages * page_size
        page = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        return int(page) * int(avail)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # /proc fallback
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:  # macOS: no avail sysconf; approximate with total/4
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) // 4
    except Exception:
        pass
    return 8 * 1024 ** 3  # 8GB default assumption


def _truncate_segments(segments, max_tokens):
    """Truncate a [(ids, mask), ...] segment list to max_tokens total."""
    out, kept = [], 0
    for seg_ids, seg_mask in segments:
        if kept >= max_tokens:
            break
        take = min(len(seg_ids), max_tokens - kept)
        out.append((seg_ids[:take], seg_mask[:take]))
        kept += take
    return out


def pack_segments_into_windows(segments, max_len):
    """Greedy-pack ordered segments (BOS, msg, msg, ...) into rows of at most
    max_len tokens, splitting an over-long segment across rows when needed.
    Order is preserved and NO token is ever discarded; the loss mask travels
    with its tokens. A conversation that fits entirely yields exactly one row
    (identical to the pre-mod behavior for normal-size conversations)."""
    rows = []
    cur_ids, cur_mask = [], []
    for seg_ids, seg_mask in segments:
        i = 0
        while i < len(seg_ids):
            space = max_len - len(cur_ids)
            if space <= 0:
                rows.append((cur_ids, cur_mask))
                cur_ids, cur_mask = [], []
                continue
            take = min(space, len(seg_ids) - i)
            cur_ids.extend(seg_ids[i:i + take])
            cur_mask.extend(seg_mask[i:i + take])
            i += take
    if cur_ids:
        rows.append((cur_ids, cur_mask))
    return rows


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        if rustbpe is None:
            raise RuntimeError(
                "rustbpe is required to TRAIN a tokenizer (pip install rustbpe), "
                "but is not installed. Inference (from_directory) works without it."
            )
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=None, num_threads=8,
                            return_segments=False):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).

        Unrestricted by design (the "no 2048 cap" mod):
        - No strict user/assistant alternation is enforced: any role order is
          rendered as-is (consecutive assistant turns, assistant-first, ...).
        - An assistant message may contain any number of text/python/python_output
          parts (multi-segment reasoning-then-answering turns are fine).
        - No fixed 2048-token cap: by default the cap is derived from the
          machine's available memory (a safe "unlimited"); an explicit
          max_tokens is still honored if a caller passes one.

        Threading: all text segments of the conversation are encoded in ONE
        tiktoken `encode_ordinary_batch` call with `num_threads` rust workers -
        the thread pool where each task is one assistant/user text segment.

        Returns:
        - ids:  list[int] token ids of the rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens the Assistant is
          trained on.
        - (only if return_segments=True) segments: list[(ids, mask)] per message,
          in order (BOS first) - used by SFT windowing to pack long conversations
          without ever discarding tokens.
        """
        messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # sometimes the first message is a system message... => merge it with the
        # next message so the template stays user/assistant only. If no user
        # message follows, render the system content as a user turn.
        if messages[0]["role"] == "system":
            # conversation surgery on a deepcopy - never mutate the caller's data
            messages = copy.deepcopy(messages)
            if len(messages) > 1 and messages[1]["role"] == "user":
                merged = self._merge_content(messages[0]["content"], messages[1]["content"])
                messages[0:2] = [{"role": "user", "content": merged}]  # system+user -> one user turn
            else:
                messages[0] = {"role": "user", "content": messages[0]["content"]}

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # ------------------------------------------------------------- plan
        # Walk the messages and build a flat encoding plan. Every plaintext
        # piece becomes one task in a single batched encode call (the rust
        # thread pool). The plan keeps (kind, ids_index, mask_val) per part so
        # the assembly below is a pure ordered paste - no cross-task state.
        plan = []          # per message: dict(role, parts=[(kind, ids_index, mask_val)...])
        texts = []         # all texts, batch-encoded in one call
        def add_text(text, kind, mask_val, parts):
            parts.append((kind, len(texts), mask_val))
            texts.append(text)

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant"):
                raise ValueError(f"Unknown message role: {role!r} (expected 'user' or 'assistant')")
            parts = []
            if role == "user":
                # strings are the norm; tolerate text-part lists for robustness
                for text in self._as_text_list(content):
                    add_text(text, "text", 0, parts)
            else:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif not isinstance(content, list):
                    raise ValueError(f"Unknown content type: {type(content)}")
                for part in content:
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        add_text(part["text"], "text", 1, parts)
                    elif ptype == "python":
                        add_text(part["text"], "python", 1, parts)
                    elif ptype == "python_output":
                        # tokens come from Python at test time => not supervised
                        add_text(part["text"], "python_output", 0, parts)
                    else:
                        raise ValueError(f"Unknown part type: {ptype}")
            plan.append(dict(role=role, parts=parts))

        # ------------------------------------------- the thread pool encode
        encoded = self.enc.encode_ordinary_batch(texts, num_threads=max(1, num_threads)) \
            if texts else []

        # ------------------------------------------------------- assemble
        ids, mask = [], []
        segments = []
        def emit(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        emit(bos, 0)
        segments.append(([bos], [0]))
        for message in plan:
            seg_start = len(ids)
            if message["role"] == "user":
                emit(user_start, 0)
                for kind, ti, mask_val in message["parts"]:
                    emit(encoded[ti], 0)
                emit(user_end, 0)
            else:
                emit(assistant_start, 0)
                for kind, ti, mask_val in message["parts"]:
                    text_ids = encoded[ti]
                    if kind == "python":
                        emit(python_start, 1); emit(text_ids, 1); emit(python_end, 1)
                    elif kind == "python_output":
                        emit(output_start, 0); emit(text_ids, 0); emit(output_end, 0)
                    else:
                        emit(text_ids, 1)
                emit(assistant_end, 1)
            segments.append((ids[seg_start:], mask[seg_start:]))

        # ------------------------------------------------- cap & truncate
        if max_tokens is None:
            max_tokens = self.auto_token_budget()
        if max_tokens is not None and len(ids) > max_tokens:
            ids, mask = ids[:max_tokens], mask[:max_tokens]
            segments = _truncate_segments(segments, max_tokens)
        if return_segments:
            return ids, mask, segments
        return ids, mask

    @staticmethod
    def _merge_content(a, b):
        """Merge two message contents (str or part-lists) into one plain string."""
        def to_text(content):
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content)
            raise ValueError(f"Unknown content type: {type(content)}")
        return to_text(a) + "\n\n" + to_text(b)

    @staticmethod
    def _as_text_list(content):
        """Accept a str or a list of {text} parts; return list[str]."""
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            return [part["text"] for part in content]
        raise ValueError(f"Unknown content type: {type(content)}")

    def auto_token_budget(self):
        """Memory-derived 'safe unlimited' cap: available RAM // ~64B per token
        (two python int lists + downstream tensor copies), floor 4096. On a
        128GB box this is hundreds of millions of tokens - effectively
        unlimited - while a tiny box still gets a sane ceiling instead of OOM."""
        try:
            avail = available_memory_bytes()
            budget = (avail * 0.5) // 64
            return max(4096, int(budget))
        except Exception:
            return 1 << 22  # ~4M tokens fallback if memory detection fails

    def render_conversations_parallel(self, conversations, max_tokens=None,
                                      num_workers=None, num_threads=8):
        """Batch-parallel version of render_conversation: a Python thread pool
        where each worker renders one conversation, and inside each worker the
        text segments go through the tiktoken rust thread pool (num_threads).
        Returns a list of (ids, mask) in input order. The rust encoder releases
        the GIL, so this genuinely parallelizes."""
        if not conversations:
            return []
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        if len(conversations) == 1 or num_workers <= 1:
            return [self.render_conversation(c, max_tokens=max_tokens, num_threads=num_threads)
                    for c in conversations]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(self.render_conversation, c, max_tokens, num_threads)
                       for c in conversations]
            return [f.result() for f in futures]

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.

        Unrestricted: if the last message is from the Assistant it is popped
        (classic SFT-style priming); if the conversation instead ends with a
        user message (or several), it is rendered as-is. Either way the
        Assistant-start token is appended so the model completes from there.
        """
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        if messages and messages[-1]["role"] == "assistant":
            messages.pop() # remove the final assistant message (in place)

        # Now tokenize the conversation (no fixed cap: memory-derived)
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# FoxSay-specific convenience functions

def get_tokenizer():
    from FoxSay.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from FoxSay.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
