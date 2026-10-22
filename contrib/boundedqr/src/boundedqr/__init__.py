"""Bounded-memory clustered QR bootstrap with explicit numerical verification."""
from .core import bootstrap,solve_perturbations,BootstrapResult,NumericalFailure
from .certification import coefficient_radius,RadiusCertificate
from .base import fn_fit
__version__='0.1.0'
__all__=['bootstrap','solve_perturbations','BootstrapResult','NumericalFailure',
         'coefficient_radius','RadiusCertificate','fn_fit']
