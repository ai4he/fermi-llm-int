"""FermiPy-facing analysis code: schema, repairs, validation and execution.

These modules know about FermiPy and the Fermi ScienceTools; nothing else in
the platform does. Components wrap them behind the contracts in
``fermi_llm.core.contracts`` so an institute can swap in a different analysis
stack without touching the web layer.
"""
