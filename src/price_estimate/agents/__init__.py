"""Agents that find online deals and decide which ones are genuinely underpriced.

Flow: ScannerAgent (RSS + LLM selection) -> EnsembleAgent (price estimate)
-> PlanningAgent (pick the biggest discount) -> MessagingAgent (push notification).
"""
