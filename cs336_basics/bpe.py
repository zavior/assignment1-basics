import heapq
import os
from collections import Counter, defaultdict
from typing import BinaryIO
import multiprocessing
import regex as re
import sys, json, pickle, resource, time

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(args) -> Counter:
    """
    Pretokenize the input file into words and count their frequencies.
    Returns a dictionary mapping each unique word to its frequency.
    """
    input_path, start, end, special_tokens = args

    word_counts = Counter()

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        text = chunk.decode("utf-8", errors="ignore")

    parts = pattern.split(text) if pattern else [text]
    for part in parts:
        words = PAT.finditer(part)
        for word in words:
            word_counts[word.group().encode("utf-8")] += 1
    return word_counts

class ReversedBytesPair:
    __slots__ = ("bytes_pair",)

    def __init__(self, bytes_pair: tuple[bytes, bytes]):
        self.bytes_pair = bytes_pair

    def __lt__(self, other):
        return self.bytes_pair > other.bytes_pair

    def __eq__(self, other):
        return self.bytes_pair == other.bytes_pair

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab : dict[int, bytes] = {i : bytes([i]) for i in range(256)}  # Initialize vocab with all single-byte tokens
    merges : list[tuple[bytes, bytes]] = []

    for special_token in special_tokens:
        # if not isinstance(special_token, str):
        #     raise ValueError(f"Special token must be a string, got {type(special_token)}")
        vocab[len(vocab)] = special_token.encode("utf-8")

    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(f, num_processes * 10, b"<|endoftext|>")

    print("finish find_chunk_boundaries, start pretokenize")

    word_counts = Counter()

    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(processes=num_processes) as pool:
        for c in pool.imap_unordered(pretokenize, tasks):
            word_counts.update(c)

    print("finish pretokenize, start train_bpe")

    word_token_ids: list[list[int]] = []
    word_freqs: list[int] = []

    for word_token_id, word_freq in word_counts.items():
        token_ids = list(word_token_id) # Convert bytes to list of ints (token IDs)
        word_token_ids.append(token_ids)
        word_freqs.append(word_freq)

    # Count the frequency of each pair of consecutive tokens in the words
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    # Map each pair to the set of words that contain it, so we can update the counts when we merge
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_token_idx, word_token_id in enumerate(word_token_ids):
        word_freq = word_freqs[word_token_idx]
        for i in range(len(word_token_id) - 1):
            pair = (word_token_id[i], word_token_id[i + 1])
            pair_counts[pair] += word_freq
            pair_to_words[pair].add(word_token_idx)

    # print(word_token_ids, word_freqs, pair_counts, pair_to_words)

    del word_counts

    heap = []
    for pair, count in pair_counts.items():
        a_token_id, b_token_id = pair
        tiebreaker = ReversedBytesPair((vocab[a_token_id], vocab[b_token_id]))
        heap.append((-count, tiebreaker, pair))  # Use negative count for max-heap behavior

    heapq.heapify(heap)

    while len(vocab) < vocab_size and heap:
        best_pair_count, _, best_pair = heapq.heappop(heap)
        best_pair_count = -best_pair_count
        if best_pair_count == 0 or pair_counts.get(best_pair, 0) != best_pair_count: 
            continue  # Skip if the count has changed since it was added to the heap

        a_token_id, b_token_id = best_pair
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[a_token_id] + vocab[b_token_id]

        merges.append((vocab[a_token_id], vocab[b_token_id]))

        pair_count_change: dict[tuple[int, int], int] = defaultdict(int)

        for word_idx in pair_to_words.pop(best_pair, set()):
            word_token_id = word_token_ids[word_idx]
            word_freq = word_freqs[word_idx]

            new_word = []
            i = 0
            flag = False
            while i < len(word_token_id):
                if i < len(word_token_id) - 1 and (word_token_id[i], word_token_id[i + 1]) == best_pair:
                    flag = True
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(word_token_id[i])
                    i += 1

            if not flag:
                continue  # No merge happened, skip updating counts

            for old_pair in zip(word_token_id, word_token_id[1:]):
                pair_count_change[old_pair] -= word_freq
                
            for pair in zip(new_word, new_word[1:]):
                pair_count_change[pair] += word_freq
                pair_to_words[pair].add(word_idx)
            word_token_ids[word_idx] = new_word

        for pair, change in pair_count_change.items():
            if change == 0:
                continue
            pair_counts[pair] += change
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair)
                pair_to_words.pop(pair, None)
            else:
                tiebreaker = ReversedBytesPair((vocab[pair[0]], vocab[pair[1]]))
                heapq.heappush(heap, (-pair_counts[pair], tiebreaker, pair))
        
        pair_counts.pop(best_pair, None)

    return vocab, merges

if __name__ == "__main__":
    print("main run")
    data_name, vocab_size, = sys.argv[1], int(sys.argv[2])
    
    input_path = "data/" + data_name + ".txt"

    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"])
    elapsed = time.perf_counter() - start

    # 峰值内存：自身 + 已结束的子进程。Linux 单位是 KB，macOS 是 bytes
    def peak_rss_bytes(who):
        r = resource.getrusage(who).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024
    peak_gb = max(peak_rss_bytes(resource.RUSAGE_SELF), peak_rss_bytes(resource.RUSAGE_CHILDREN)) / 1e9

    # 序列化
    with open(f"out/{data_name}_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    
    longest = sorted(vocab.values(), key=len, reverse=True)[:5]
    print(f"耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)，峰值内存 {peak_gb:.2f} GB")
    print(f"词表 {len(vocab)}，merges {len(merges)}")
    print("最长的 5 个 token：", [t.decode("utf-8", errors="replace") for t in longest], [len(t) for t in longest])
