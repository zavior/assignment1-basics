import os, sys, pickle
import regex as re
from multiprocessing import Pool
from functools import lru_cache

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    """
    A simple tokenizer class that can be used to tokenize text based on a given vocabulary and merges.
    """
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]  | None = None,
                  cache_size: int = 300_000):
        self.vocab = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.ranked_merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.pattern = re.compile("|".join(re.escape(t) for t in self.special_tokens))
        # self._hot_cache = {v: k for token_id, token_bytes in vocab.items()}
        self._lru_cache = lru_cache(maxsize=cache_size)(self._encode_word)

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
            tmp = pickle.load(f)
            vocab = tmp["vocab"]
            merges = tmp["merges"]
        return cls(vocab, merges, special_tokens)
    
    def _encode_word(self, word: bytes) -> tuple[int]:
        """
        Encode a single word into a list of token IDs based on the vocabulary and merges.
        """
        # Start with the initial tokens (characters)
        tokens = [bytes([b]) for b in word]

        # print(f"Initial tokens for word {word}: {tokens}")

        # Apply merges

        # naive implementation of BPE merge operations
        # for merge in self.merges:
        #     merged_token = merge[0] + merge[1]
        #     # if merged_token in self.vocab: # necessary?
        #     # print(f"Current tokens: {tokens}")
        #     # print(f"Applying merge: {merge} -> {merged_token}")
        #     new_tokens = []
        #     i = 0
        #     while i < len(tokens):
        #         if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == (merge[0], merge[1]):
        #             new_tokens.append(merged_token)
        #             i += 2
        #         else:
        #             new_tokens.append(tokens[i])
        #             i += 1

        #     tokens = new_tokens

        # optimized implementation of BPE merge operations
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            best_rank = -1
            best_pair = None
            for pair in pairs:
                pair_rank = self.ranked_merges.get(pair, -1)
                if pair_rank != -1 and (best_rank == -1 or pair_rank < best_rank):
                    best_rank = pair_rank
                    best_pair = pair

            if not best_pair:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tuple(self.bytes_to_id[t] for t in tokens)

    def encode_without_special_tokens(self, text : str, start: int, end: int) -> list[int]:
        text_id_seq = []
        words = PAT.finditer(text[start:end])
        for word in words:
            # print(f"Encoding word: {word.group()}")
            token_bytes = word.group().encode("utf-8")
            # if token_bytes in self.cache:
            #     text_id_seq.extend(self.cache[token_bytes])
            # else:
            #     word_id_seq = self._encode_word(token_bytes)
            #     text_id_seq = text_id_seq + word_id_seq
            #     self.cache[token_bytes] = word_id_seq
            text_id_seq.extend(self._lru_cache(token_bytes))
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

    def _find_chunk_boundaries(self, file_path: str | os.PathLike, desired_num_chunk: int) -> list[int]:
        if self.special_tokens:
            split_pat = re.compile(b"|".join(re.escape(token.encode("utf-8")) for token in self.special_tokens))
            overlap = max(len(token.encode("utf-8")) for token in self.special_tokens)         
        else:
            split_pat = re.compile(rb"\s(?=\S)")
            overlap = 1

        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            chunk_size = file_size // desired_num_chunk
            boundaries = [i * chunk_size for i in range(desired_num_chunk + 1)]
            boundaries[-1] = file_size

            mini_chunk_size = 4096

            for bi in range(1, len(boundaries) - 1):
                pos = boundaries[bi]
                
                while True:
                    f.seek(pos)
                    mini_chunk = f.read(mini_chunk_size + overlap)
                    if mini_chunk == b"":
                        boundaries[bi] = file_size
                        break

                    match = split_pat.search(mini_chunk)
                    if match:
                        boundaries[bi] = pos + match.start()
                        break
                    if len(mini_chunk) < overlap:
                        boundaries[bi] = file_size
                        break
                    pos += mini_chunk_size

        return sorted(set(boundaries))
    
    def _encode_for_parallel(self, file_path: str | os.PathLike, start: int, end: int) -> list[int]:
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return self.encode(chunk)

    def encode_file_parallel(self, file_path: str | os.PathLike, num_processes: int = 1) -> list[int]:
        boundaries = self._find_chunk_boundaries(file_path, num_processes * 20)

        ret = []
    
        with Pool(num_processes) as pool:
            results = pool.starmap(self._encode_for_parallel, [(file_path, start, end) for start, end in zip(boundaries, boundaries[1:])])
            for r in results:
                ret.extend(r)

        return ret

    # lru_cache 包装的绑定方法不能 pickle；macOS 默认用 spawn 启动子进程，需要以下两个方法
    def __getstate__(self):
        s = self.__dict__.copy(); s.pop("_lru_cache", None); return s
    
    def __setstate__(self, s):
        self.__dict__.update(s); self._lru_cache = lru_cache(maxsize=300_000)(self._encode_word)

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
 
    # ---------- 3. encode_file_parallel：结果应与 encode(全文) 一致 ----------
    import tempfile, pathlib, random
 
    EOT = "<|endoftext|>"
 
    # 字节级词表：256 个单字节 + 若干 merge（含中文多字节、"\n\n"），保证任意 UTF-8 文本都能编码
    byte_vocab = {i: bytes([i]) for i in range(256)}
    byte_merges = []
    for a, b in [(b't', b'h'), (b'th', b'e'), (b' ', b't'), (b' t', b'he'), (b' ', b'c'),
                 (b' c', b'a'), (b' ca', b't'), (b'\n', b'\n'),
                 ("你".encode()[:2], "你".encode()[2:]), ("好".encode()[:2], "好".encode()[2:])]:
        byte_merges.append((a, b))
        byte_vocab[len(byte_vocab)] = a + b
    byte_vocab[len(byte_vocab)] = EOT.encode()
    byte_vocab[len(byte_vocab)] = b"<|pad|>"
 
    btok_sp = get_tokenizer(byte_vocab, byte_merges, [EOT])
    btok_multi = get_tokenizer(byte_vocab, byte_merges, [EOT, "<|pad|>"])
    btok_plain = get_tokenizer(byte_vocab, byte_merges)
 
    def make_corpus(n_docs: int, seed: int = 0) -> str:
        """随机生成 n_docs 篇文档，用 EOT 分隔；内容覆盖空格、换行、空行、中文、标点、数字"""
        rng = random.Random(seed)
        words = ["the", "cat", "ate", " ", "\n", "\n\n", "你好", "世界", "!", "123", "it's", "  ", "\t"]
        docs = ["".join(rng.choice(words) + rng.choice(["", " "]) for _ in range(rng.randint(0, 40)))
                for _ in range(n_docs)]
        return EOT.join(docs)
 
    tmpdir = tempfile.TemporaryDirectory()
 
    def write_tmp(name: str, text: str) -> pathlib.Path:
        p = pathlib.Path(tmpdir.name) / name
        p.write_bytes(text.encode("utf-8"))  # 用 bytes 写，避免 Windows 换行转换
        return p
 
    def run_parallel(name, tok, text, num_processes):
        path = write_tmp(f"{abs(hash(name))}.txt", text)
        try:
            got = tok.encode_file_parallel(path, num_processes=num_processes)
        except NotImplementedError:
            got = "NotImplementedError"
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        check(f"parallel | {name} | np={num_processes}", got, tok.encode(text))
        if isinstance(got, list):
            check(f"par-rt   | {name} | np={num_processes}", tok.decode(got), text)
 
    big = make_corpus(300, seed=1)
    parallel_cases = [
        # 含 special token
        ("sp | small corpus",                 btok_sp,    make_corpus(5)),
        ("sp | big corpus",                   btok_sp,    big),
        ("sp | special at start & end",       btok_sp,    EOT + "the cat" + EOT),
        ("sp | consecutive specials",         btok_sp,    "the" + EOT * 5 + "cat"),
        ("sp | only special",                 btok_sp,    EOT),
        ("sp | blank lines around special",   btok_sp,    "the\n\n" + EOT + "\n\n\ncat\n\n"),
        ("sp | multibyte around special",     btok_sp,    "你好" + EOT + "世界" + EOT + "你好"),
        ("sp | single doc, no special inside", btok_sp,   "the cat ate\n\nthe cat " * 50),
        ("sp | multiple special tokens",      btok_multi, "the<|pad|>cat" + EOT + "ate<|pad|><|pad|>" + EOT),
        # 不含 special token（tokenizer 本身没配置）
        ("no-sp | plain text",                btok_plain, "the cat ate\n\nthe 你好 cat\n" * 30),
        ("no-sp | eot is plain text here",    btok_plain, "the" + EOT + "cat"),
        # 边界
        ("edge | empty file",                 btok_sp,    ""),
        ("edge | single char",                btok_sp,    "t"),
    ]
    for name, tok, text in parallel_cases:
        for np_ in (1, 2, 4):
            run_parallel(name, tok, text, np_)
 
    # 进程数远大于文档数
    run_parallel("edge | processes >> docs", btok_sp, "the" + EOT + "cat", 16)
 
    # 不同进程数结果完全一致（确定性）
    p_big = write_tmp("big.txt", big)
    try:
        results = [btok_sp.encode_file_parallel(p_big, num_processes=n) for n in (1, 2, 3, 4, 8)]
        check("parallel | deterministic across num_processes", all(r == results[0] for r in results), True)
    except Exception as e:
        check("parallel | deterministic across num_processes", f"{type(e).__name__}", "no error")
 
    # 路径支持 str 和 pathlib.Path
    for kind, arg in [("str", str(p_big)), ("Path", p_big)]:
        try:
            got = btok_sp.encode_file_parallel(arg, num_processes=2)
            check(f"parallel | path as {kind}", got, btok_sp.encode(big))
        except Exception as e:
            check(f"parallel | path as {kind}", f"{type(e).__name__}", "no error")
 
    # 返回类型 list[int]
    try:
        got = btok_sp.encode_file_parallel(p_big, num_processes=2)
        check("parallel | returns list[int]",
              isinstance(got, list) and all(isinstance(x, int) for x in got), True)
    except Exception as e:
        check("parallel | returns list[int]", f"{type(e).__name__}", "no error")
 
    # 文件不存在
    try:
        btok_sp.encode_file_parallel(pathlib.Path(tmpdir.name) / "missing.txt", num_processes=2)
        check("parallel | missing file -> FileNotFoundError", "no error", "FileNotFoundError")
    except FileNotFoundError:
        check("parallel | missing file -> FileNotFoundError", "FileNotFoundError", "FileNotFoundError")
    except Exception as e:
        check("parallel | missing file -> FileNotFoundError", type(e).__name__, "FileNotFoundError")
 
    tmpdir.cleanup()
 
    print("====" * 10)
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
