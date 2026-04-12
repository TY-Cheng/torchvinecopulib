from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BiCopDiagnostics"]


@dataclass
class BiCopDiagnostics:
    itp_failures_l: int
    itp_failures_r: int
    bisect_refinements_l: int
    bisect_refinements_r: int
    fallback_to_indep_l: int
    fallback_to_indep_r: int
    max_abs_hfunc_error_l: float
    max_abs_hfunc_error_r: float
