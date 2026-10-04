"""
SPECTRA - Execution Layer

Translates a router ExecutionPlan into a real ONNX Runtime session.

This is the module that makes the headline claim true: the WorkloadRouter
decides which silicon should run a task, and this layer is what actually
builds the InferenceSession with those execution providers.

Without it, `selected_provider` is decoration.
"""

from core.execution.session_builder import (
    ProviderResolution,
    ExecutionSession,
    build_providers,
    create_session,
    provider_device_map,
)

__all__ = [
    "ProviderResolution",
    "ExecutionSession",
    "build_providers",
    "create_session",
    "provider_device_map",
]
