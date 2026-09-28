"""Horsea: fast-memory adaptation and fast->slow consolidation for flow-matching policies."""
import os

# LIBERO reads its asset/bddl/init paths from $LIBERO_CONFIG_PATH/config.yaml. The user's
# ~/.libero points at a deleted conda env, so every entrypoint uses the project-local config.
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".libero")
)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
