import os, sys, time, resource
from collections import Counter
from cs336_basics.tokenizer import Tokenizer

if __name__ == "__main__":
    for dataset_name in ["owt_train", "TinyStoriesV2-GPT4-train"]:
        tokenizer = Tokenizer.from_cpickle(f"out/{dataset_name}_bpe.pkl", ["<|endoftext|>"])
        EOT = "<|endoftext|>"
        docs, buf = [], []

        start = time.perf_counter()

        # 峰值内存：自身 + 已结束的子进程。Linux 单位是 KB，macOS 是 bytes
        def peak_rss_bytes(who):
            r = resource.getrusage(who).ru_maxrss
            return r if sys.platform == "darwin" else r * 1024
        peak_gb = max(peak_rss_bytes(resource.RUSAGE_SELF), peak_rss_bytes(resource.RUSAGE_CHILDREN)) / 1e9

        with open(f"data/{dataset_name}.txt", encoding="utf-8", newline="") as f:
            for line in f:
                buf.append(line)
                if EOT in line:
                    docs.extend(d for d in "".join(buf).split(EOT)[:-1] if d.strip())  # 按 EOT 切出完整文档
                    buf = ["".join(buf).split(EOT)[-1]]
                    if len(docs) >= 1000:
                        break
        docs = docs[:1000]          # 或 random.sample(all_docs, 10)

        elapsed = time.perf_counter() - start

        n_bytes = n_tokens = 0
        for d in docs:
            ids = tokenizer.encode(d)              # 整篇编码，不按行
            assert tokenizer.decode(ids) == d
            n_bytes += len(d.encode("utf-8"))      # 真实字节数
            n_tokens += len(ids)
        print(f"{dataset_name}: {len(docs)} docs, {n_bytes} bytes, {n_tokens} tokens, "
            f"{n_bytes / n_tokens:.2f} bytes/token, elapsed: {elapsed:.2f} s, peak: {peak_gb:.2f} GB, "
            f"{n_bytes / elapsed:.2f} bytes/second")
        