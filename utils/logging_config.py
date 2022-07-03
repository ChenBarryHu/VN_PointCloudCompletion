LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "basic": {"format": "%(name)s - %(levelname)s - %(message)s"},
        "extended": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "train": {"level": "DEBUG", "propagate": True,},
        "dataset": {"level": "DEBUG", "propagate": True,},
    },
    "root": {"level": "DEBUG", "handlers": ["console"],},
}
