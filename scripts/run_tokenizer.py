"""
把原始文本数据集流式 tokenize 成 uint16 raw 二进制文件（读取：np.memmap(path, dtype=np.uint16, mode="r")）。

用法（每个数据集单独跑一次，峰值内存统计才准确）：
    python encode_dataset.py TinyStoriesV2-GPT4-train
    python encode_dataset.py owt_train
"""
import os
import sys
import time
import resource
from itertools import islice

import numpy as np

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
BATCH = 1 << 20  # 每批 100 万个 id（uint16 约 2MB）


def peak_rss_gb() -> float:
    # 本进程从启动至今的峰值内存。Linux 单位是 KB，macOS 是 bytes
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / 1e9


def encode_to_bin(tokenizer: Tokenizer, txt_path: str, bin_path: str) -> int:
    """边读边编码边写；先写 .tmp，成功后再改名，避免中断留下半个文件。返回 token 数。"""
    tmp_path = bin_path + ".tmp"
    total = 0
    # newline="" 保留原始换行（\r\n 不转成 \n），与按二进制读取的并行版结果一致
    with open(txt_path, "r", encoding="utf-8", newline="") as f, open(tmp_path, "wb") as out:
        it = tokenizer.encode_iterable(f)
        while (arr := np.fromiter(islice(it, BATCH), dtype=np.uint16)).size:  # id > 65535 会抛 OverflowError
            arr.tofile(out)
            total += arr.size
    os.replace(tmp_path, bin_path)
    return total


def verify(tokenizer: Tokenizer, txt_path: str, bin_path: str, total: int) -> None:
    ids = np.memmap(bin_path, dtype=np.uint16, mode="r")
    assert len(ids) == total == os.path.getsize(bin_path) // 2, "token 数与文件大小不一致"

    # 开头一段能还原回原文
    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        head = f.read(10_000)
    decoded = tokenizer.decode(ids[:5_000].tolist())
    n = min(len(decoded), len(head)) - 10  # 末尾可能截断在多字节字符中间，留点余量
    assert n > 0 and decoded[:n] == head[:n], "开头解码结果与原文不一致"

    # EOT 个数 == 原文中 <|endoftext|> 出现次数（流式数一遍原文，内存有限）
    eot_id = tokenizer.bytes_to_id[EOT.encode("utf-8")]
    n_eot_bin = int(sum((ids[i:i + (1 << 26)] == eot_id).sum() for i in range(0, len(ids), 1 << 26)))
    n_eot_txt = 0
    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        for line in f:  # EOT 不含换行，不会跨行
            n_eot_txt += line.count(EOT)
    assert n_eot_bin == n_eot_txt, f"EOT 数不一致: bin={n_eot_bin}, txt={n_eot_txt}"
    print(f"  verify OK: {total:,} tokens, {n_eot_bin:,} EOT")


def main(dataset_name: str) -> None:
    txt_path = f"data/{dataset_name}.txt"
    bin_path = f"out/{dataset_name}_tokens.bin"
    tokenizer = Tokenizer.from_cpickle(f"out/{dataset_name}_bpe.pkl", [EOT])

    print(f"Processing dataset: {dataset_name}")
    start = time.perf_counter()
    total = encode_to_bin(tokenizer, txt_path, bin_path)
    elapsed = time.perf_counter() - start

    n_bytes = os.path.getsize(txt_path)
    print(f"  {total:,} tokens, {elapsed:.1f}s ({n_bytes / 1e6 / elapsed:.2f} MB/s), "
          f"compression {n_bytes / total:.2f} bytes/token, peak {peak_rss_gb():.2f} GB")

    verify(tokenizer, txt_path, bin_path, total)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {sys.argv[0]} <dataset_name> [<dataset_name> ...]")
    for name in sys.argv[1:]:  # 也可一次传多个；但峰值内存会是累计的最大值
        main(name)