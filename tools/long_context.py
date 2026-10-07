#!/usr/bin/env python3
"""Long-context quality for the YaRN ramp decision: several needles at different depths, and key=value retrieval over
a long synthetic document, at 128k / 256k / 512k / 1M-token prompts, scored by exact match, written as a JSON report
so two server configurations can be compared A/B.

Usage:
  tools/long_context.py [--label L] [--sizes 128k,256k,512k,1M] [--tests needles,kv] [--depths 0.02,0.25,0.5,0.75,0.98]
                        [--keys 8] [--seed 1] [--out FILE]
  tools/long_context.py --compare A.json B.json
  Sizes: k = 1,000 tokens, M = 1,000,000. API_URL / PORT / MODEL as in client.py; thinking off, greedy.

Per size, two prompts (one prefill each):
  needles  generated prose with one "vault code" sentence at each depth; one question asks for every code.
  kv       a document of `key = value` lines; one question asks for --keys of them, spread over the document.
Every asked item is scored exact (the reply's `name = value` line for that name has exactly the value) and present (the
value appears anywhere in the reply). Prompts are seeded: the same settings give the same prompts (sha256 recorded per
test), so --compare can tell whether A and B saw identical prompts. A size above the server's window is recorded as
skipped, never scored. The report is rewritten after every test (a long run that dies keeps what it measured).
Exit (run): 0 every test ran, 1 some test errored, 2 none ran. Exit (compare): 0 comparable, 3 prompts differ or missing.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import sys

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

OVERHEAD = 400                           # question + chat template + reply room kept below the size, in tokens
SYL = ("mar bel cor dun fen hal is kel lor mon nor ost pel quin ros sel tor ul ven wil yar zen ash brin cal").split()


def place(rng: random.Random) -> str:
    return "".join(rng.choice(SYL) for _ in range(3)).capitalize() + rng.choice(("ford", "mere", "holt", "wick", "vale"))


class KvLines:
    """`key = value` lines with unique keys (a fresh instance per document)."""

    def __init__(self):
        self.seen = set()
        self.facts = []

    def __call__(self, rng: random.Random) -> str:
        while True:
            key = f"{rng.choice(client._ADJS)}-{rng.choice(client._NOUNS)}-{rng.randint(1000, 9999)}"
            if key not in self.seen:
                break
        self.seen.add(key)
        value = f"{rng.randint(10000, 99999)}-{rng.choice(client._NOUNS)}"
        self.facts.append((key, value))
        return f"{key} = {value}"


def _norm(s: str) -> str:
    s = s.strip().strip("`*_\"'").strip()
    return re.sub(r"[\s.,;:`*_\"']+$", "", s).strip().lower()


def parse_pairs(reply: str) -> dict:
    """`name = value` (or `name: value`) lines of a reply, names lowercased, list markers and quotes dropped."""
    out = {}
    for line in reply.splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line)
        m = re.match(r"^(.+?)\s*(?:=|:|->|—|–)\s*(.+)$", line)
        if m:
            out.setdefault(_norm(m.group(1)), _norm(m.group(2)))
    return out


def score(reply: str, asked: list) -> list:
    pairs, low = parse_pairs(reply), reply.lower()
    items = []
    for it in asked:
        got = pairs.get(it["key"].lower())
        items.append(dict(it, answer=got, exact=got == it["expected"].lower(), present=it["expected"].lower() in low))
    return items


def build_needles(size: int, seed: int, depths: list, cpt: float):
    rng = random.Random(f"needles-{seed}-{size}")
    names, codes = [], []
    while len(names) < len(depths):
        n = place(rng)
        if n not in names:
            names.append(n)
            codes.append(f"{rng.randint(1000, 9999)}-{rng.choice(client._NOUNS)}")
    hay = client.filler_units(client.sentence, size - OVERHEAD - 25 * len(depths), seed * 1000 + size % 997, cpt)
    asked = []
    for i in sorted(range(len(depths)), key=lambda j: -depths[j]):   # deepest first, so earlier indexes stay put
        at = min(len(hay), max(0, int(len(hay) * depths[i])))
        hay.insert(at, f"The vault code for {names[i]} is {codes[i]}.")
    for i in range(len(depths)):
        asked.append({"depth": depths[i], "key": names[i], "expected": codes[i]})
    order = [a["key"] for a in asked]
    rng.shuffle(order)
    question = ("\n\nThe text above gives vault codes for some places. Give the vault code of each of these places, "
                "one per line, written as `place = code`, and nothing else: " + ", ".join(order) + ".")
    return " ".join(hay) + question, asked


def build_kv(size: int, seed: int, keys: int, cpt: float):
    gen = KvLines()
    lines = client.filler_units(gen, size - OVERHEAD - 20 * keys, seed * 1000 + 7 + size % 991, cpt, joiner="\n")
    n = len(gen.facts)
    picks = sorted({min(n - 1, int((i + 0.5) * n / keys)) for i in range(keys)})
    asked = [{"depth": round(i / n, 4), "key": gen.facts[i][0], "expected": gen.facts[i][1]} for i in picks]
    order = [a["key"] for a in asked]
    random.Random(f"kv-order-{seed}-{size}").shuffle(order)
    head = "Below is a list of facts, one per line, written as `key = value`.\n\n"
    question = ("\n\nGive the value of each of these keys from the list above, one per line, written as "
                "`key = value`, and nothing else: " + ", ".join(order) + ".")
    return head + "\n".join(lines) + question, asked


def run_tests(a) -> int:
    sizes = [client.parse_size(s) for s in a.sizes.split(",") if s.strip()]
    depths = [float(d) for d in a.depths.split(",") if d.strip()]
    kinds = [k.strip() for k in a.tests.split(",") if k.strip()]
    for k in kinds:
        if k not in ("needles", "kv"):
            raise client.ApiError(f"unknown test {k!r} (needles, kv)")
    if any(not 0 <= d <= 1 for d in depths):
        raise client.ApiError("depths are fractions from 0 to 1")
    model = client.model_id()
    win = client.window()
    cpt_prose, m1 = client.calibrate(client.sentence, a.seed)
    cpt_kv, m2 = client.calibrate(KvLines(), a.seed, 1500)
    out = a.out or f"long_context-{a.label}.json"
    report = {"tool": "long_context", "format": 1, "label": a.label,
              "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "server": {"url": client.API_URL, "model": model, "window": win},
              "settings": {"sizes": sizes, "tests": kinds, "depths": depths, "keys": a.keys, "seed": a.seed,
                           "max_tokens": a.max_tokens, "chars_per_token": {"prose": cpt_prose, "kv": cpt_kv},
                           "chars_per_token_measured": bool(m1 and m2)},
              "tests": []}
    if not (m1 and m2):
        print(f"note: sized at {cpt_prose}/{cpt_kv} characters a token (no /tokenize answer, or CHARS_PER_TOKEN set)",
              file=sys.stderr)
    ran = errored = 0
    for size in sizes:
        for kind in kinds:
            t = {"size": size, "kind": kind}
            if win is not None and size + a.max_tokens > win:
                t.update(status="skipped", reason=f"{size} + {a.max_tokens} reply tokens exceed the server's window {win}")
                print(f"{size:>8} {kind:<7} skipped: {t['reason']}", flush=True)
                report["tests"].append(t)
                write(out, report)
                continue
            if kind == "needles":
                prompt, asked = build_needles(size, a.seed, depths, cpt_prose)
            else:
                prompt, asked = build_kv(size, a.seed, a.keys, cpt_kv)
            t["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            t["prompt_chars"] = len(prompt)
            try:
                r = client.chat([{"role": "user", "content": prompt}], max_tokens=a.max_tokens, thinking=False,
                                temperature=0.0, seed=1234, timeout=a.timeout)
            except client.ApiError as exc:
                errored += 1
                t.update(status="error", reason=str(exc), items=[dict(x, answer=None, exact=False, present=False)
                                                                  for x in asked])
                print(f"{size:>8} {kind:<7} error: {exc}", flush=True)
                report["tests"].append(t)
                write(out, report)
                continue
            ran += 1
            items = score(r.content, asked)
            t.update(status="ran", prompt_tokens=r.prompt_tokens, prefill_seconds=r.prefill_seconds,
                     seconds=round(r.seconds, 3), finish_reason=r.finish_reason, token_sha=r.token_sha,
                     exact=sum(x["exact"] for x in items), present=sum(x["present"] for x in items), of=len(items),
                     items=items, reply=r.content[:4000])
            pre = f"{r.prefill_seconds:.1f} s" if r.prefill_seconds is not None else "n/a"
            print(f"{size:>8} {kind:<7} prompt {r.prompt_tokens} tok, prefill {pre}, exact {t['exact']}/{len(items)}, "
                  f"present {t['present']}/{len(items)}"
                  + "".join(f"\n           {x['depth']:<6} {x['key']:<28} want {x['expected']:<16} got {x['answer']!s:<16}"
                            f" {'exact' if x['exact'] else ('present' if x['present'] else 'MISS')}" for x in items),
                  flush=True)
            report["tests"].append(t)
            write(out, report)
    scored = [t for t in report["tests"] if t["status"] == "ran"]
    report["summary"] = {"ran": ran, "errors": errored,
                         "skipped": sum(t["status"] == "skipped" for t in report["tests"]),
                         "exact": sum(t["exact"] for t in scored), "present": sum(t["present"] for t in scored),
                         "of": sum(t["of"] for t in scored),
                         "by_size": {str(s): {t["kind"]: (f"{t['exact']}/{t['of']}" if t["status"] == "ran" else t["status"])
                                              for t in report["tests"] if t["size"] == s} for s in sizes}}
    write(out, report)
    s = report["summary"]
    print(f"long_context {a.label}: {s['ran']} tests ran, {s['errors']} errors, {s['skipped']} skipped; exact "
          f"{s['exact']}/{s['of']}, present {s['present']}/{s['of']}; report {out}", flush=True)
    if ran == 0:
        return client.CANNOT_RUN
    return client.FAIL if errored else client.PASS


def write(path: str, report: dict) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def compare(pa: str, pb: str) -> int:
    def load(p):
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError) as exc:
            raise client.ApiError(f"cannot read the report {p}: {exc}") from None
        if r.get("tool") != "long_context":
            raise client.ApiError(f"{p} is not a long_context report")
        return r
    A, B = load(pa), load(pb)
    ta = {(t["size"], t["kind"]): t for t in A["tests"]}
    tb = {(t["size"], t["kind"]): t for t in B["tests"]}
    fair = True
    la, lb = A.get("label", "A"), B.get("label", "B")
    print(f"{'size':>8} {'test':<7} {la[:18]:>18} {lb[:18]:>18}  prefill A / B      prompts")
    tot = {"A": [0, 0], "B": [0, 0]}
    for key in sorted(set(ta) | set(tb)):
        x, y = ta.get(key), tb.get(key)

        def cell(t):
            if t is None:
                return "missing"
            return f"{t['exact']}/{t['of']}" if t["status"] == "ran" else t["status"]
        same = x is not None and y is not None and x.get("prompt_sha256") and x.get("prompt_sha256") == y.get("prompt_sha256")
        both = x is not None and y is not None and x["status"] == "ran" and y["status"] == "ran"
        if not same or not both:
            fair = False
        if both and same:
            tot["A"][0] += x["exact"]; tot["A"][1] += x["of"]
            tot["B"][0] += y["exact"]; tot["B"][1] += y["of"]

        def pre(t):
            v = t.get("prefill_seconds") if t else None
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        print(f"{key[0]:>8} {key[1]:<7} {cell(x):>18} {cell(y):>18}  {pre(x):>6} / {pre(y):<8}  "
              + ("same" if same else "DIFFER (not an A/B)"))
    print(f"total on identical prompts both ran: {la} {tot['A'][0]}/{tot['A'][1]}, {lb} {tot['B'][0]}/{tot['B'][1]}")
    if not fair:
        print("compare: some tests are missing, did not run on both sides, or saw different prompts: those rows are "
              "not counted in the total")
        return client.UNCHECKED
    return client.PASS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", default="run")
    p.add_argument("--sizes", default="128k,256k,512k,1M")
    p.add_argument("--tests", default="needles,kv")
    p.add_argument("--depths", default="0.02,0.25,0.5,0.75,0.98")
    p.add_argument("--keys", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=7200, help="seconds a request may take (1M-token prefills are long)")
    p.add_argument("--out", help="report path (default long_context-<label>.json)")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = p.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.keys < 1:
        p.error("--keys is at least 1")
    return run_tests(a)


if __name__ == "__main__":
    client.run(main)
