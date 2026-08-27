import json
import logging
import sys
import os


class JsonFormatter(logging.Formatter):
    """ELK-friendly JSON formatter — enabled when VSECTOR_LOG_JSON=1."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info and record.exc_info[0]:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    use_json = os.getenv("VSECTOR_LOG_JSON", "0") == "1"
    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        handler.setFormatter(JsonFormatter())
        fmt = None
    else:
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[handler],
        force=True,
    )
    if use_json:
        # re-apply json formatter after basicConfig creates handlers
        for h in logging.getLogger().handlers:
            h.setFormatter(JsonFormatter())
