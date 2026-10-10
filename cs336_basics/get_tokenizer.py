import os, sys, cpickle
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    """
    A simple tokenizer class that can be used to tokenize text based on a given vocabulary and merges.
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        # if self.special_tokens:
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(t) for t in self.special_tokens))
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """ 
        Class method
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        pass

    @classmethod
    def from_cpickle(cls, cpickle_filepath, special_tokens=None):
        """
        Class method
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        cpickle_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(cpickle_filepath, "rb") as f:
            vocab, merges = cpickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def _encode_word(self, word: bytes) -> list[int]:
        """
        Encode a single word into a list of token IDs based on the vocabulary and merges.
        """
        # Start with the initial tokens (characters)
        tokens = [bytes([b]) for b in word]

        # print(f"Initial tokens for word {word}: {tokens}")

        # Apply merges
        for merge in self.merges:
            merged_token = merge[0] + merge[1]
            # if merged_token in self.vocab: # necessary?
            # print(f"Current tokens: {tokens}")
            # print(f"Applying merge: {merge} -> {merged_token}")
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == (merge[0], merge[1]):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return [self.bytes_to_id[t] for t in tokens]

    def encode_without_special_tokens(self, text : str, start: int, end: int) -> list[int]:
        text_id_seq = []
        words = PAT.finditer(text[start:end])
        for word in words:
            # print(f"Encoding word: {word.group()}")
            token_bytes = word.group().encode("utf-8")
            if token_bytes in self.cache:
                text_id_seq.extend(self.cache[token_bytes])
            else:
                word_id_seq = self._encode_word(token_bytes)
                text_id_seq.extend(word_id_seq)
                self.cache[token_bytes] = word_id_seq
        return text_id_seq

    def encode(self, text: str) -> list[int]:
        text_id_seq = []
        parts = self.pattern.finditer(text) if self.special_tokens else []
        last_end = 0
        for part in parts:

            # print(f"Encoding part: {text[last_end:part.start()]} and special token: {part.group()}")

            text_id_seq.extend(self.encode_without_special_tokens(text, last_end, part.start()))
            special_token_bytes = part.group().encode("utf-8")
            if special_token_bytes in self.bytes_to_id:
                text_id_seq.append(self.bytes_to_id[special_token_bytes])

            last_end = part.end()

        # print(f"Encoding remaining part: {text[last_end:]}")

        text_id_seq.extend(self.encode_without_special_tokens(text, last_end, len(text)))
        return text_id_seq

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in tokens).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> any:
        blank_pattern = re.compile(r"\s+(?!\S)")
        buffer = str()
        first_line = True
        for line in iterable:
            if first_line:
                first_line = False
                buffer = line
            else:
                if blank_pattern.fullmatch(line[0]):
                    buffer += line
                else: # if the line does not start with a blank, yield the encoded buffer and start a new buffer
                    yield from self.encode(buffer[:-1])
                    buffer = buffer[-1] + line
        yield from self.encode(buffer) # yield the last buffer

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> any:
    return Tokenizer(vocab, merges, special_tokens)

if __name__ == "__main__":
    import io
 
    # 小词表：原有 token + 换行/字母 b + 双换行合并 + 两个 special token
    vocab = {
        0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
        6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at',
        11: b'<|endoftext|>',
        12: b'\n', 13: b'b', 14: b'\n\n', 15: b' b',
        16: b'<|pad|>', 17: b'<|endoftext|><|endoftext|>',
    }
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't'),
              (b'\n', b'\n'), (b' ', b'b')]
 
    passed = failed = 0
 
    def check(name, got, expected):
        global passed, failed
        if got == expected:
            passed += 1
            print(f"[PASS] {name}")
        else:
            failed += 1
            print(f"[FAIL] {name}\n       got:      {got}\n       expected: {expected}")
 
    # ---------- 1. encode / decode 基础用例 ----------
    tok_sp = get_tokenizer(vocab, merges, ["<|endoftext|>"])
    tok_plain = get_tokenizer(vocab, merges)  # 不带 special token
 
    encode_cases = [
        # (name, tokenizer, text, expected_ids)
        ("basic sentence",            tok_sp,    "the cat ate",                   [9, 7, 1, 5, 10, 3]),
        ("single byte",               tok_sp,    "t",                             [5]),
        ("empty string",              tok_sp,    "",                              []),
        ("special token w/ spaces",   tok_sp,    "the cat <|endoftext|> ate",     [9, 7, 1, 5, 0, 11, 10, 3]),
        ("special token no spaces",   tok_sp,    "the cat<|endoftext|>ate",       [9, 7, 1, 5, 11, 1, 5, 3]),
        ("only special token",        tok_sp,    "<|endoftext|>",                 [11]),
        ("adjacent special tokens",   tok_sp,    "<|endoftext|><|endoftext|>",    [11, 11]),
        ("special at start and end",  tok_sp,    "<|endoftext|>the<|endoftext|>", [11, 9, 11]),
        ("double newline",            tok_sp,    "a\n\nb",                        [1, 12, 12, 13]),
        ("no special: plain text",    tok_plain, "the cat ate",                   [9, 7, 1, 5, 10, 3]),
    ]
    for name, tok, text, expected in encode_cases:
        ids = tok.encode(text)
        check(f"encode   | {name}", ids, expected)
        check(f"roundtrip| {name}", tok.decode(ids), text)
 
    # 未注册为 special 时，"<|endoftext|>" 按普通文本切分（这里会因字节不在词表而 KeyError）
    try:
        tok_plain.encode("<|endoftext|>")
        check("no special: <|endoftext|> treated as text -> KeyError", "no error", "KeyError")
    except KeyError:
        check("no special: <|endoftext|> treated as text -> KeyError", "KeyError", "KeyError")
 
    # 重叠 special token：更长者优先
    tok_overlap = get_tokenizer(vocab, merges, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    check("encode   | overlapping specials, longest wins",
          tok_overlap.encode("<|endoftext|><|endoftext|>"), [17])
 
    # 多个 special token
    tok_multi = get_tokenizer(vocab, merges, ["<|endoftext|>", "<|pad|>"])
    check("encode   | multiple special tokens",
          tok_multi.encode("the<|pad|>cat<|endoftext|>"), [9, 16, 2, 1, 5, 11])
 
    # ---------- 2. encode_iterable：StringIO 输入，结果应与 encode(全文) 一致 ----------
    iterable_cases = [
        # 不含 special token
        ("no-sp | single line",                 "the cat ate"),
        ("no-sp | multi lines",                 "the cat\nate the\ncat"),
        ("no-sp | trailing newline",            "the cat\n"),
        ("no-sp | line starting with space",    "the\n cat\n ate"),
        ("no-sp | blank line between",          "a\n\nb"),
        ("no-sp | multiple blank lines",        "the\n\n\n\ncat"),
        ("no-sp | only newlines",               "\n\n\n"),
        ("no-sp | empty input",                 ""),
        # 含 special token
        ("sp    | special mid line",            "the cat <|endoftext|> ate"),
        ("sp    | special at line start",       "the cat\n<|endoftext|>ate"),
        ("sp    | special at line end",         "the cat<|endoftext|>\nate"),
        ("sp    | special on its own line",     "the\n<|endoftext|>\ncat"),
        ("sp    | adjacent specials",           "a<|endoftext|><|endoftext|>b"),
        ("sp    | special + blank lines",       "the<|endoftext|>\n\n\ncat\n"),
        ("sp    | only special",                "<|endoftext|>"),
    ]
    for name, text in iterable_cases:
        expected = tok_sp.encode(text)
        got = list(tok_sp.encode_iterable(io.StringIO(text)))
        check(f"iterable | {name}", got, expected)
        check(f"iter-rt  | {name}", tok_sp.decode(got), text)
 
    # tokenizer 未配置 special token 时也走 encode_iterable
    for name, text in [("plain tok | multi lines", "the cat\nate\n\nthe"),
                       ("plain tok | indented",    "a\n b\n\n a")]:
        got = list(tok_plain.encode_iterable(io.StringIO(text)))
        check(f"iterable | {name}", got, tok_plain.encode(text))
 
    # encode_iterable 应该是惰性的生成器
    import types
    check("iterable | returns generator",
          isinstance(tok_sp.encode_iterable(io.StringIO("a")), types.GeneratorType), True)
 
    print("====" * 10)
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
