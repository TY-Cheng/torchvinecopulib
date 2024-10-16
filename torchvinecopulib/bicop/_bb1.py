import torch
from ._archimedean import BiCopArchimedean

class Bb1(BiCopArchimedean):

    _PAR_MIN, _PAR_MAX = (0, 1.01), (6.99, 6.99)
    

    @staticmethod
    def generator(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return pow(pow(obs[:,[0]], -par[0]) -1, par[1])
        
    @staticmethod
    def generator_inv(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return pow(pow(obs[:,[0]], 1 / par[1]) + 1, -1 / par[0])
    
    
    @staticmethod
    def generator_derivative(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        res = -delta * theta * pow(obs[:,[0]], -(1 + theta))
        return res * pow(pow(obs[:,[0]], -theta) - 1, delta - 1)
    
    @staticmethod
    def pars2tau(par: tuple[float]) -> torch.Tensor:
        return 1 - 2 / (par[1] * (par[0] + 2))
    
    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        
        theta = par[0]
        delta = par[1]
        
        t1 = pow(obs[:,[0]], -theta)
        t2 = t1 - 1
        t3 = pow(t2, delta)
        t16 = 1/obs[:,[0]]
        t17 = 1/t2
        t38 = t1*t16
        t39 = t38*t17
        t4 = pow(obs[:,[1]], -theta)
        t5 = t4 - 1
        t6 = pow(t5, delta)
        t7 = t3 + t6
        t9 = pow(t7, 1/delta)
        t10 = 1 + t9
        t12 = pow(t10, -1/theta)
        t13 = t12*t9
        t20 = 1/t10
        t24 = t9*t9
        t25 = t12*t24
        t27 = 1/obs[:,[1]]
        t29 = 1/t5
        t32 = t7*t7
        t33 = 1/t32
        t34 = t10*t10
        t36 = t33/t34
        t43 = t4*theta
        t59 = t43*t27*t29
        
        return t25 * t6 * t27 * t4 * t29 * t36 * t3 * t39 - t13 * t6 * t43 * t27 * t29 * t33 * t3 * t38 * t17 * t20 + t13 * t3 * t38 * t17 * t33 * t20 * t6 * delta * t59 + t25 * t3 * t39 * t36 * t6 * t59
    

