#!/usr/bin/env python3
"""Image prompt reuse on the Qwen TensorFold server.

The running :8888 server does not keep a prompt that contains a picture, so every
later turn prefills the whole conversation again. The branch `image-prompt-reuse`
lifts that. This check is the receipt. It does not build or restart anything.

When you can take :8888 down, on spark1:

    cd ~/src/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold
    git checkout image-prompt-reuse
    ./scripts/prepare.sh --rebuild          # new image only; the live container stays up
    # copy the image to the worker the way you usually do, then:
    ./stop.sh && ./start.sh
    python3 tools/image_prompt_reuse.py 12k 3

Pass: a follow-up that still carries the same picture resumes at least 85% of the
prompt, and a second conversation with the same words but different pixels does
not resume through that picture (the bulk of the prompt is after the picture, so
a false hit would cache most of it).

Exit 0 pass, 1 fail, 2 could not run, 3 the server reported no cached-token count.
"""
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


def part_image(blob: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(blob), "detail": "low"}}


def user(blob: bytes, before: str, after: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": before},
        part_image(blob),
        {"type": "text", "text": after},
    ]}


def fmt(v, spec="{:.2f} s"):
    return spec.format(v) if v is not None else "n/a"


def one(messages) -> "client.Reply":
    return client.chat(messages, max_tokens=32, thinking=False, temperature=0.0)


def share_of(reply) -> tuple:
    cached, prompt = reply.cached_tokens, reply.prompt_tokens
    if cached is None or not prompt:
        return None, cached, prompt
    return cached / prompt, cached, prompt


def main() -> int:
    size = client.parse_size(sys.argv[1] if len(sys.argv) > 1 else "12k")
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if size < 8000:
        print("image_prompt_reuse: size must be at least 8k tokens (the cache ignores shorter prompts)")
        return client.CANNOT_RUN
    if turns < 1:
        print("image_prompt_reuse: turns is at least 1")
        return client.CANNOT_RUN
    cpt, _ = client.calibrate(client.sentence, 11)
    red, blue = png(64, 64, (220, 30, 30)), png(64, 64, (30, 30, 220))
    system = {"role": "system", "content": "You are a careful assistant. Answer in one short sentence. "
              + text(400, 1, cpt)}
    # The picture sits near the start. The long text is after it, so a false reuse of another
    # picture would cache most of the prompt, and a real one stops before the picture.
    before = "Look at the picture, then the field log.\n\n"
    after = text(size, 2, cpt)
    convo = [system, user(red, before, after)]
    try:
        r = one(convo)
    except client.ApiError as e:
        print(f"image_prompt_reuse: CANNOT RUN: {e}")
        return client.CANNOT_RUN
    share, cached, prompt = share_of(r)
    if share is None:
        print(f"turn 0: prompt {prompt} cached {cached} prefill {fmt(r.prefill_seconds)} wall {r.seconds:.1f} s")
        print("image_prompt_reuse: UNCHECKED: the server reports no cached-token count")
        return client.UNCHECKED
    print(f"turn 0 (cold): prompt {prompt} cached {cached} ({share:.1%}) prefill {fmt(r.prefill_seconds)} "
          f"wall {r.seconds:.1f} s", flush=True)
    shares = []
    for turn in range(1, turns + 1):
        convo += [{"role": "assistant", "content": r.content or "noted"},
                  {"role": "user", "content": f"Note {turn}: {text(80, 100 + turn, cpt)} What is one new detail?"}]
        r = one(convo)
        share, cached, prompt = share_of(r)
        if share is None:
            print("image_prompt_reuse: UNCHECKED: a follow-up reported no cached-token count")
            return client.UNCHECKED
        shares.append(share)
        print(f"turn {turn} (same picture still in the history): prompt {prompt} cached {cached} ({share:.1%}) "
              f"prefill {fmt(r.prefill_seconds)} wall {r.seconds:.1f} s", flush=True)
    other = [system, user(blue, before, after)]
    o = one(other)
    oshare, ocached, oprompt = share_of(o)
    if oshare is None:
        print("image_prompt_reuse: UNCHECKED: the other-picture prompt reported no cached-token count")
        return client.UNCHECKED
    print(f"other picture, same words: prompt {oprompt} cached {ocached} ({oshare:.1%}) "
          f"prefill {fmt(o.prefill_seconds)} wall {o.seconds:.1f} s", flush=True)
    worst = min(shares)
    if worst < 0.85:
        print(f"image_prompt_reuse: FAIL: a same-picture follow-up resumed only {worst:.1%} (want >= 85%)")
        return client.FAIL
    if oshare > 0.5:
        print(f"image_prompt_reuse: FAIL: a different picture resumed {oshare:.1%} of the same words "
              "(want <= 50%; the picture is near the start, so a real miss stops there)")
        return client.FAIL
    print(f"image_prompt_reuse: PASS: same-picture follow-ups resumed >= 85% (worst {worst:.1%}); "
          f"a different picture resumed {oshare:.1%}")
    return client.PASS


if __name__ == "__main__":
    try:
        sys.exit(main())
    except client.ApiError as e:
        print(f"image_prompt_reuse: CANNOT RUN: {e}")
        sys.exit(client.CANNOT_RUN)
