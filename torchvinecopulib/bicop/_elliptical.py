from ._abc import BiCopAbstract
import torch


class BiCopElliptical(BiCopAbstract):
    # Joe 2014 page 164
    @staticmethod
    def rho2tau_0(rho: torch.Tensor) -> torch.Tensor:
        """ρ to Kendall's τ"""
        return rho.asin() * 0.6366197723675814

    @staticmethod
    def tau2rho_0(tau: torch.Tensor) -> torch.Tensor:
        """Kendall's τ to ρ"""
        return (tau * 1.5707963267948966).sin()
