from math import asin, sin

from ._abc import BiCopAbstract


class BiCopElliptical(BiCopAbstract):
    # Joe 2014 page 164
    @staticmethod
    def rho2tau_0(rho: float) -> float:
        """ρ to Kendall's τ"""
        return asin(rho) * 0.6366197723675814

    @staticmethod
    def tau2rho_0(tau: float) -> float:
        """Kendall's τ to ρ"""
        return sin(tau * 1.5707963267948966)
