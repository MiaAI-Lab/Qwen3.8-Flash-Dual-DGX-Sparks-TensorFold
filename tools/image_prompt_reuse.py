#!/usr/bin/env python3
"""Image prompt reuse: a prompt that contains a picture resumes, and a different picture does not.

Usage: tools/image_prompt_reuse.py [size] [turns] [--min-reuse 0.85]
  size   tokens of text after the picture (default 12k; '8k' accepted), turns the follow-up turns (default 3).
  API_URL / PORT / MODEL as in client.py; thinking off, greedy, fixed seed.
The picture sits near the start and the long text is after it, so a false reuse of another picture would
cache most of the prompt. A same-picture follow-up must resume at least --min-reuse of its prompt. The same
prompt sent again must resume, and that reply must equal the fresh prefill (text, finish reason, and the
server's token sha when both replies report one): the comparison tools/exact.py makes.
The same words with different pixels must not resume through that picture.
Size is at least 8k tokens: shorter prompts are not kept.
Exit 0 pass, 1 fail, 2 could not run, 3 the server reported no cached-token count.
"""
import argparse
import hashlib
import struct
import sys
import zlib

sys.dont_write_bytecode = True
import client  # noqa: E402


def png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def data_url(blob: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(blob).decode()


def text(tokens: int, seed: int, cpt: float) -> str:
    return " ".join(client.filler_units(client.sentence, tokens, seed, cpt))


def user(blob: bytes, before: str, after: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": before},
        {"type": "image_url", "image_url": {"url": data_url(blob), "detail": "low"}},
        {"type": "text", "text": after},
    ]}


def fmt(v, spec="{:.2f} s"):
    return spec.format(v) if v is not None else "n/a"


def one(messages) -> client.Reply:
    return client.chat(messages, max_tokens=32, thinking=False, temperature=0.0, seed=4321)


def share_of(reply: client.Reply):
    cached, prompt = reply.cached_tokens, reply.prompt_tokens
    if cached is None or not prompt:
        return None, cached, prompt
    return cached / prompt, cached, prompt


def text_sha(r: client.Reply) -> str:
    """The same reply digest as tools/exact.py."""
    blob = r.content + "\x00" + r.reasoning + "\x00" + "".join(
        f"{c['name']}({c['arguments']})" for c in r.tool_calls)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def same(a: client.Reply, b: client.Reply) -> bool:
    """exact.py's comparison: text and finish reason, plus token sha when both replies report one."""
    if text_sha(a) != text_sha(b) or a.finish_reason != b.finish_reason:
        return False
    if a.token_sha is not None and b.token_sha is not None and a.token_sha != b.token_sha:
        return False
    return True


def line(label: str, reply: client.Reply, share) -> str:
    cached, prompt = reply.cached_tokens, reply.prompt_tokens
    shown = f"{share:.1%}" if share is not None else "not reported"
    tok = reply.token_sha or "-"
    return (f"{label}: prompt {prompt} cached {cached} ({shown}) prefill {fmt(reply.prefill_seconds)} "
            f"wall {reply.seconds:.1f} s text {text_sha(reply)} token {tok}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("size", nargs="?", default="12k")
    p.add_argument("turns", nargs="?", type=int, default=3)
    p.add_argument("--min-reuse", type=float, default=0.85)
    a = p.parse_args()
    size = client.parse_size(a.size)
    if size < 8000:
        print("image_prompt_reuse: size must be at least 8k tokens (shorter prompts are not kept)")
        return client.CANNOT_RUN
    if a.turns < 1:
        print("image_prompt_reuse: turns is at least 1")
        return client.CANNOT_RUN
    cpt, _ = client.calibrate(client.sentence, 11)
    red, blue = png(64, 64, (220, 30, 30)), png(64, 64, (30, 30, 220))
    system = {"role": "system", "content": "You are a careful assistant. Answer in one short sentence. "
              + text(400, 1, cpt)}
    before = "Look at the picture, then the field log.\n\n"
    after = text(size, 2, cpt)
    convo = [system, user(red, before, after)]
    fresh = one(convo)
    share, _, _ = share_of(fresh)
    if share is None:
        print(line("turn 0", fresh, share))
        print("image_prompt_reuse: UNCHECKED: the server reports no cached-token count")
        return client.UNCHECKED
    print(line("turn 0 (fresh)", fresh, share), flush=True)
    resumed = one(convo)
    rshare, _, _ = share_of(resumed)
    if rshare is None:
        print("image_prompt_reuse: UNCHECKED: the resumed prompt reported no cached-token count")
        return client.UNCHECKED
    print(line("turn 0 again (resume)", resumed, rshare), flush=True)
    shares = [rshare]
    if not same(fresh, resumed):
        print("image_prompt_reuse: FAIL: the resumed reply differs from the fresh prefill")
        print(f"  fresh  : {fresh.content[:160]!r}")
        print(f"  resumed: {resumed.content[:160]!r}")
        return client.FAIL
    if fresh.token_sha is None or resumed.token_sha is None:
        print("  note: token sha was not reported on both replies; compared the text, as tools/exact.py does")
    follow = resumed
    history = list(convo)
    for turn in range(1, a.turns + 1):
        history += [{"role": "assistant", "content": follow.content or "noted"},
                    {"role": "user", "content": f"Note {turn}: {text(80, 100 + turn, cpt)} What is one new detail?"}]
        follow = one(history)
        share, _, _ = share_of(follow)
        if share is None:
            print("image_prompt_reuse: UNCHECKED: a follow-up reported no cached-token count")
            return client.UNCHECKED
        shares.append(share)
        print(line(f"turn {turn} (same picture still in the history)", follow, share), flush=True)
    other = one([system, user(blue, before, after)])
    oshare, _, _ = share_of(other)
    if oshare is None:
        print("image_prompt_reuse: UNCHECKED: the other-picture prompt reported no cached-token count")
        return client.UNCHECKED
    print(line("other picture, same words", other, oshare), flush=True)
    worst = min(shares)
    if worst < a.min_reuse:
        print(f"image_prompt_reuse: FAIL: a same-picture prompt resumed only {worst:.1%} "
              f"(want >= {a.min_reuse:.0%})")
        return client.FAIL
    if oshare > 0.5:
        print(f"image_prompt_reuse: FAIL: a different picture resumed {oshare:.1%} of the same words "
              "(want <= 50%; the picture is near the start, so a real miss stops there)")
        return client.FAIL
    print(f"image_prompt_reuse: PASS: same-picture prompts resumed >= {a.min_reuse:.0%} (worst {worst:.1%}) "
          f"and the resumed reply equalled the fresh prefill; a different picture resumed {oshare:.1%}")
    return client.PASS


if __name__ == "__main__":
    client.run(main)
