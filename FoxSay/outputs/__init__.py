"""
FoxSay.outputs -- the model's feedback channels to the outside world.

  web_api      : tiny HTTP portal (stdlib http.server) for text in/out and
                 media upload -- the model's mouth and ears on the web.
  media_inbox  : incoming image / video / document reception: saves uploads,
                 registers them in an index file, and exposes them to the
                 training inputs (FoxSay.inputs) as a feedback loop.

Both use a shared Message format so inputs and outputs plug together:

    {"role": "user"|"model", "kind": "text"|"image"|"video"|"document",
     "content": str, "path": str|None, "ts": float}
"""

from . import media_inbox, web_api
from .media_inbox import MediaInbox
from .web_api import WebPortal

__all__ = ["media_inbox", "web_api", "MediaInbox", "WebPortal"]
