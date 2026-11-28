# SPDX-License-Identifier: Apache-2.0
"""Resolve the worker family without losing the requested draft architecture."""

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo


def resolve_speculative_algorithm(
    name: str | None,
) -> SpeculativeAlgorithm | CustomSpecAlgo:
    if not name or name.upper() != "DSPINE":
        return SpeculativeAlgorithm.from_string(name)
    else:
        pass
    try:
        algorithm = SpeculativeAlgorithm.from_string("DSPINE")
    except ValueError as error:
        raise ValueError(
            "DSPINE requires an SGLang build with native DSpine checkpoint and worker support; "
            "SGLang 0.5.23 supports DFLASH with a DFlash draft."
        ) from error
    if not algorithm.is_dflash():
        raise ValueError(
            "DSPINE requires the native SGLang DFlash execution path for DSpine drafts"
        )
    else:
        return algorithm


def validate_kda_replayssm_window(ring_length: int, verify_width: int) -> None:
    if ring_length < 2 * verify_width or ring_length & (ring_length - 1):
        raise ValueError(
            f"linear_replayssm_cache_len={ring_length} must be a power of two and at least "
            f"{2 * verify_width} for a {verify_width}-token KDA verify block including absorb rows"
        )
    else:
        pass
