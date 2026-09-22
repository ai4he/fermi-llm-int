"""Application services: the orchestration core code performs.

A service composes components; it never implements a feature that a
component could own. ``chat`` runs generation plus the guardrail stack,
``runs`` drives review and execution, ``exports`` serves artifacts.
"""
