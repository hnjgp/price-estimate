from __future__ import annotations

import logging

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

logger = logging.getLogger("price_estimate.agents")


class Agent:
    """Base class: gives every agent a name and a colour in the shared log."""

    name: str = "Agent"
    color: str = WHITE

    def log(self, message: str) -> None:
        logger.info(f"{BG_BLACK}{self.color}[{self.name}] {message}{RESET}")
