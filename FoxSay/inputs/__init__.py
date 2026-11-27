"""
FoxSay.inputs -- the model's five growth channels (training inputs).

Every channel is a FEEDER: an iterable of text that the shared trainer
consumes one batch at a time.  No channel writes a big dataset to disk --
everything is streamed.

  teacher_api  : multi-AI conversation distillation (any OpenAI-compatible API)
  web_crawler  : internet knowledge feeding (BFS crawl -> clean -> stream)
  cli_trainer  : human-in-the-loop chat training (CLI conversations become
                 training data immediately)
  av_learner   : audio/video watching (subtitle/metadata text extraction;
                 inject an ASR transcript function for full speech)
  ebook_reader : book / e-book parsing (txt native, epub via zipfile+xml,
                 pdf via an injectable extractor)

Shared trainer: FoxSay.inputs.trainer.train_on_feeders(...) runs the
streaming loop and periodically saves brain region shards.
"""

from . import trainer
from .teacher_api import TeacherStream
from .web_crawler import WebFeeder
from .cli_trainer import CLIChatTrainer
from .av_learner import AVFeeder
from .ebook_reader import EbookFeeder

__all__ = [
    "trainer", "TeacherStream", "WebFeeder", "CLIChatTrainer",
    "AVFeeder", "EbookFeeder",
]
