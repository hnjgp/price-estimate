"""Wires the agents together and keeps a memory of deals already surfaced."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from price_estimate.agents.base import BG_BLUE, RESET, WHITE
from price_estimate.agents.deals import Opportunity
from price_estimate.config import settings

_LOGGING_READY = False


def init_logging() -> None:
    """Configure root logging once (the original added a new handler on every run)."""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [Agents] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _LOGGING_READY = True


class Memory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.memory_file)
        self.items: list[Opportunity] = self.read()

    def read(self) -> list[Opportunity]:
        if not self.path.exists():
            return []
        try:
            return [Opportunity(**o) for o in json.loads(self.path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError, ValueError):
            logging.warning("memory file %s is corrupt; starting empty", self.path)
            return []

    def add(self, opportunity: Opportunity) -> None:
        self.items.append(opportunity)
        self.path.write_text(
            json.dumps([o.model_dump() for o in self.items], indent=2), encoding="utf-8"
        )


class DealAgentFramework:
    def __init__(self, planner=None, memory: Memory | None = None, autonomous: bool = False):
        init_logging()
        self.memory = memory or Memory()
        self.autonomous = autonomous
        self.planner = planner

    def log(self, message: str) -> None:
        logging.info(f"{BG_BLUE}{WHITE}[Agent Framework] {message}{RESET}")

    def build_planner(self):
        from price_estimate.agents.ensemble_agent import EnsembleAgent
        from price_estimate.agents.messaging import MessagingAgent
        from price_estimate.agents.scanner import ScannerAgent
        from price_estimate.rag import get_collection

        self.log("initializing agents")
        try:
            collection = get_collection()
        except Exception as error:
            self.log(f"vector store unavailable ({error}); frontier model disabled")
            collection = None
        scanner, pricer, messenger = ScannerAgent(), EnsembleAgent(collection), MessagingAgent()
        if self.autonomous:
            from price_estimate.agents.autonomous import AutonomousPlanningAgent

            return AutonomousPlanningAgent(scanner, pricer, messenger)
        from price_estimate.agents.planning import PlanningAgent

        return PlanningAgent(scanner, pricer, messenger)

    def run(self) -> list[Opportunity]:
        if self.planner is None:
            self.planner = self.build_planner()
        result = self.planner.plan(memory=self.memory.items)
        self.log(f"run complete: {result}")
        if result:
            self.memory.add(result)
        return self.memory.items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one deal-hunting cycle")
    parser.add_argument(
        "--autonomous", action="store_true", help="let the LLM orchestrate the tools"
    )
    DealAgentFramework(autonomous=parser.parse_args().autonomous).run()
