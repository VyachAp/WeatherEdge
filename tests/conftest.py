"""Shared pytest fixtures.

The trading-knob unit tests read the live ``src.config.settings`` singleton,
which loads from the production ``.env``. When ``.env`` sets a knob to a
non-default value (e.g. ``PROBABILITY_MIN_ENTRY_PRICE=0.80``,
``MAX_POSITION_PCT=0.10``, ``NEAR_PEAK_FLOOR_UP_ENABLED=True``), tests that
assert the *code-default* behaviour break even though the code is correct.

This autouse fixture resets the deployment-sensitive knobs back to their
``Settings`` model-field defaults for every test, so the suite exercises CODE
behaviour independent of the deployed config. A test that intentionally opts
into a non-default value still wins: its own ``monkeypatch.setattr`` runs inside
the test body, after this fixture.
"""

import pytest

from src.config import Settings, settings

# Knobs the production .env overrides but which unit tests assume at their code
# defaults. Keep this list tight — only knobs that (a) .env sets to a
# non-default and (b) some test asserts the default of. Values come from the
# Settings model so they stay in sync with the code defaults automatically.
_ENV_SENSITIVE_DEFAULTS = (
    "PROBABILITY_MIN_ENTRY_PRICE",
    "MAX_POSITION_PCT",
    "NEAR_PEAK_FLOOR_UP_ENABLED",
    "NEAR_PEAK_FLOOR_STAKE_USD",
    "LOCK_CONVICTION_SIZING_ENABLED",
    "NEAR_LOCK_CONVICTION_SIZING_ENABLED",
)


@pytest.fixture(autouse=True)
def _reset_env_sensitive_settings(monkeypatch):
    for name in _ENV_SENSITIVE_DEFAULTS:
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)
