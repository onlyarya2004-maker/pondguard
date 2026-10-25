"""PondGuard: an agentic early-warning advisor for shrimp-pond disease.

A supervisor plans each run and routes four specialist agents (Monitor,
Diagnostician, Advisor, Critic); risky actions pause for human approval.
Everything runs locally with small language models in LM Studio.
"""

from pondguard.paths import configure_local_env

__version__ = "0.1.0"

configure_local_env()
