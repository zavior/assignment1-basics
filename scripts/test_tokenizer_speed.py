import os, sys
from collections import Counter
from cs336_basics.tokenizer import Tokenizer

if __name__ == "__main__":
    for dataset_name in ["owt_train", "TinyStoriesV2-GPT4-train"]:
        tokenizer = Tokenizer.from_cpickle(f"out/{dataset_name}_bpe.pkl", ["<|endoftext|>"])
        EOT = "<|endoftext|>"
        docs, buf = [], []
        with open(f"data/{dataset_name}.txt", encoding="utf-8", newline="") as f:
            for line in f:
                buf.append(line)
                if EOT in line:
                    docs.extend(d for d in "".join(buf).split(EOT)[:-1] if d.strip())  # 按 EOT 切出完整文档
                    buf = ["".join(buf).split(EOT)[-1]]
                    if len(docs) >= 100:
                        break
        docs = docs[:100]          # 或 random.sample(all_docs, 10)

        n_bytes = n_tokens = 0
        for d in docs:
            ids = tokenizer.encode(d)              # 整篇编码，不按行
            assert tokenizer.decode(ids) == d
            n_bytes += len(d.encode("utf-8"))      # 真实字节数
            n_tokens += len(ids)
        print(f"{dataset_name}: {len(docs)} docs, {n_bytes} bytes, {n_tokens} tokens, "
            f"{n_bytes / n_tokens:.2f} bytes/token")