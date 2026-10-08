"""Token frequencies (Qwen tokenizer) over local text (GLM drafter corpora: prompts + decoded outputs) -> draft vocab."""
import json, glob, collections, sys
from tokenizers import Tokenizer
qt = Tokenizer.from_file("models/qwen38fn-exl3-4.05bpw/tokenizer.json")
gt = Tokenizer.from_file(glob.glob("models/hf/hub/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/*/tokenizer.json")[0])
cnt = collections.Counter(); held = collections.Counter()
files = sorted(glob.glob("glm53-engine/bench/opd/think*/gen.jsonl"))
n = 0
for f in files:
    for i, line in enumerate(open(f)):
        d = json.loads(line)
        texts = [d.get("prompt", "")[:200000], gt.decode(d.get("output_ids", []), skip_special_tokens=False)]
        target = held if i % 10 == 0 else cnt            # every 10th record held out for coverage
        for t in texts:
            ids = qt.encode(t, add_special_tokens=False).ids
            target.update(ids); n += len(ids)
print("tokens", n, "distinct", len(cnt))
order = [t for t, _ in cnt.most_common()]
tot_h = sum(held.values())
for V in (16384, 32768, 49152, 65536):
    s = set(order[:V]) | set(range(248044, 248320))
    cov = sum(c for t, c in held.items() if t in s) / tot_h
    print(f"V={V}: held-out coverage {cov:.5f}")
V = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
keep = set(cnt) | set(held) | set(range(248044, 248320))          # every token seen + all special tokens
for t in range(248044):                                             # fill with the lowest ids (most frequent BPE merges)
    if len(keep) >= V: break
    keep.add(t)
keep = sorted(keep)[:V] if len(keep) > V else sorted(keep)
assert len(keep) == V, len(keep)
tot_h = sum(held.values())
print(f"V={V}: held-out coverage {sum(c for t, c in held.items() if t in set(keep)) / tot_h:.5f}")
json.dump(keep, open(f"models/draft_vocab_{V}.json", "w"))
print("saved", len(keep))
