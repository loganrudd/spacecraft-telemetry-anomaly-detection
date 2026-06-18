"""ISS Live channel registry.

Encodes the locked 26-PUI channel map from .claude/rules/iss.md.
Subsystem names (power, solar_array, thermal, attitude) are the same strings
used by the HPO grouping logic so they flow through to Ray Tune unchanged.

Usage:
    from spacecraft_telemetry.ingest.iss_channels import (
        ISS_CHANNELS,
        VALIDATION_CHANNELS,
        subscription_items,
    )
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class ChannelMeta(NamedTuple):
    description: str
    unit: str
    subsystem: str


# ---------------------------------------------------------------------------
# Telemetry channels (26 PUIs, 4 subsystems)
# ---------------------------------------------------------------------------
# Subsystem names match the existing HPO grouping convention.
# SARJ angles and battery SOC are intentionally excluded (see iss.md).

ISS_CHANNELS: dict[str, ChannelMeta] = {
    # -- power: PV voltage + current (orbital charge/discharge cycle) ----------
    "P4000001": ChannelMeta("Solar Array 2A Voltage", "V", "power"),
    "P4000002": ChannelMeta("Solar Array 2A Current", "A", "power"),
    "S4000001": ChannelMeta("Solar Array 1A Voltage", "V", "power"),
    "S4000002": ChannelMeta("Solar Array 1A Current", "A", "power"),
    "P6000001": ChannelMeta("Solar Array 4B Voltage", "V", "power"),
    "P6000002": ChannelMeta("Solar Array 4B Current", "A", "power"),
    "S6000001": ChannelMeta("Solar Array 3B Voltage", "V", "power"),
    "S6000002": ChannelMeta("Solar Array 3B Current", "A", "power"),
    # -- solar_array: Beta Gimbal Assembly pointing angles (smooth ramps) ------
    "P4000007": ChannelMeta("BGA 2A Position", "°", "solar_array"),
    "P4000008": ChannelMeta("BGA 4A Position", "°", "solar_array"),
    "P6000007": ChannelMeta("BGA 4B Position", "°", "solar_array"),
    "P6000008": ChannelMeta("BGA 2B Position", "°", "solar_array"),
    "S4000007": ChannelMeta("BGA 1A Position", "°", "solar_array"),
    "S4000008": ChannelMeta("BGA 3A Position", "°", "solar_array"),
    "S6000007": ChannelMeta("BGA 3B Position", "°", "solar_array"),
    "S6000008": ChannelMeta("BGA 1B Position", "°", "solar_array"),
    # -- thermal: Coolant loop temps + Thermal Radiator Rotary Joint positions -
    "S1000003": ChannelMeta("Loop A (Stbd) PM Out Temp", "°C", "thermal"),
    "P1000003": ChannelMeta("Loop B (Port) PM Out Temp", "°C", "thermal"),
    "S0000001": ChannelMeta("Stbd TRRJ Position", "°", "thermal"),
    "S0000002": ChannelMeta("Port TRRJ Position", "°", "thermal"),
    # -- attitude: Station orientation + CMG wheel speeds ---------------------
    "USLAB000018": ChannelMeta("LVLH Quaternion 0", "—", "attitude"),
    "USLAB000019": ChannelMeta("LVLH Quaternion 1", "—", "attitude"),
    "USLAB000020": ChannelMeta("LVLH Quaternion 2", "—", "attitude"),
    "USLAB000021": ChannelMeta("LVLH Quaternion 3", "—", "attitude"),
    "Z1000009": ChannelMeta("CMG 1 Wheel Speed", "rpm", "attitude"),
    "Z1000010": ChannelMeta("CMG 2 Wheel Speed", "rpm", "attitude"),
}

# ---------------------------------------------------------------------------
# Context items — subscribed and archived, not telemetry channels
# ---------------------------------------------------------------------------
# TIME_000001: Greenwich Mean Time; used downstream for LOS derivation.
# USLAB000086: ISS Station Mode; operational context.

CONTEXT_ITEMS: list[str] = ["TIME_000001", "USLAB000086"]

# ---------------------------------------------------------------------------
# Validation set — first 6 channels to prove end-to-end (iss.md §Phase 12)
# ---------------------------------------------------------------------------

VALIDATION_CHANNELS: list[str] = [
    "S1000003",    # thermal — cleanest signal
    "P1000003",    # thermal — second loop
    "P4000007",    # solar_array — smooth ramp, strongly orbital
    "S4000007",    # solar_array — opposite-truss beta angle
    "P4000001",    # power — voltage, surfaces eclipse-flatline handling early
    "USLAB000018", # attitude — quaternion component, bounded
]

# ---------------------------------------------------------------------------
# Valid subsystem names
# ---------------------------------------------------------------------------

SUBSYSTEMS: frozenset[str] = frozenset(m.subsystem for m in ISS_CHANNELS.values())


def subscription_items(channel_set: Literal["validation", "all"]) -> list[str]:
    """Return the sorted list of Lightstreamer item names to subscribe.

    Includes the requested telemetry PUIs plus CONTEXT_ITEMS.  Context items
    are always included so the collector archives TIME_000001 for downstream
    LOS derivation regardless of which channel_set is active.

    Args:
        channel_set: ``"validation"`` for the 6-channel Phase 12 validation
            set; ``"all"`` for all 26 telemetry channels.

    Returns:
        Sorted list of item names (telemetry + context).
    """
    puis = list(VALIDATION_CHANNELS) if channel_set == "validation" else list(ISS_CHANNELS)
    return sorted(set(puis) | set(CONTEXT_ITEMS))
