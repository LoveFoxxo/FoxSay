"""
test_io_portal.py -- inputs package (ebook/av/trainer smoke) and outputs
package (web portal text round-trip + media upload) on tiny models.
Run:  python tests/test_io_portal.py
"""

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tiny_model():
    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=2, n_embd=32, n_head=2, n_kv_head=2,
                      vocab_size=50257, sequence_len=128))
    m.init_weights()
    return m


def test_ebook_and_av():
    from FoxSay.inputs.ebook_reader import EbookFeeder, read_epub
    from FoxSay.inputs.av_learner import AVFeeder, read_srt
    with tempfile.TemporaryDirectory() as d:
        # txt
        p = os.path.join(d, "book.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("Chapter one. " * 200)
        texts = list(EbookFeeder([p]))
        assert texts and "Chapter one" in texts[0]
        # srt
        s = os.path.join(d, "cap.srt")
        with open(s, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nhello world\n\n")
        got = list(AVFeeder([s]))
        assert "hello world" in got[0], got
        # pdf without extractor -> graceful message
        pdf = os.path.join(d, "x.pdf")
        open(pdf, "w").write("%PDF-1.4 fake")
        assert "pdf needs" in list(EbookFeeder([pdf]))[0]
    print("ebook/av feeders: PASS")


def test_trainer_smoke():
    import tiktoken
    from FoxSay.inputs import trainer
    tok = tiktoken.get_encoding("gpt2")
    m = tiny_model()  # vocab 50257 == gpt2 tokenizer
    with tempfile.TemporaryDirectory() as d:
        steps = trainer.train_on_feeders(
            m, tok, feeders=[iter(["hello world " * 40] * 5)], steps=2,
            out_dir=d, n_shards=2, save_every=1, batch=1, lr=1e-4,
            vocab_size=50257)
        assert steps >= 1
        assert os.listdir(d), "shards must be written"
    print("inputs.trainer streaming loop: PASS")


def test_web_portal():
    import tiktoken
    from FoxSay.outputs import WebPortal
    tok = tiktoken.get_encoding("gpt2")
    m = tiny_model()  # vocab 50257 == gpt2 tokenizer
    with tempfile.TemporaryDirectory() as d:
        portal = WebPortal(m, tok, port=18731, save_dir=d, max_new=8)
        portal.start_background()
        time.sleep(0.8)
        base = "http://127.0.0.1:18731"
        # text round-trip
        req = urllib.request.Request(base + "/message",
                                     data=json.dumps({"text": "hi"}).encode(),
                                     headers={"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert "reply" in reply
        # history
        hist = json.loads(urllib.request.urlopen(base + "/history", timeout=10).read())
        assert len(hist["history"]) >= 2
        # upload
        req = urllib.request.Request(
            base + "/upload?name=pic.png&desc=test%20image",
            data=b"\x89PNG fake", method="POST")
        up = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert up["kind"] == "image"
        items = json.loads(urllib.request.urlopen(base + "/items", timeout=10).read())
        assert len(items["items"]) == 1
    print("outputs.web_api text+upload: PASS")


if __name__ == "__main__":
    test_ebook_and_av()
    test_trainer_smoke()
    test_web_portal()
    print("ALL IO PORTAL TESTS PASSED")
