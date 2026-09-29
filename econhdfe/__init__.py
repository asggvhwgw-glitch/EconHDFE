from .compute.runtime import configure_numba_runtime
configure_numba_runtime()

from .errors import (
    EconHDFEError, InputError, DataTypeError, ShapeError, MissingDataError, InvalidWeightError,
    SpecificationError, IdentificationError, UnderidentifiedError, CollinearityError,
    ConvergenceError, NumericalError, LinearAlgebraError, DivergenceError, InferenceError, BootstrapError,
)
from .frontend import VariableRole, PreflightIssue, PreflightReport, preflight_dataframe
from .config import HDFEConfig, InferenceConfig, ExecutionConfig
from .compute.context import ExecutionContext
from .reporting import PublicationResult, publication_output
from .api import olshdfe, ppmlhdfe, ivhdfe, ivppmlhdfe, reghdfe, ivreghdfe, PPMLHDFE, PPMLConfig, PPMLResult, IVPPMLHDFE, IVPPMLConfig, IVPPMLResult, SPJPanel, SPJResult, ivppml_spj, SPJBootstrapResult, ivppml_spj_bootstrap
from .hdfe import (
    HDFEAbsorber, HDFEConvergenceError, GroupIndividualAbsorber, GroupIndividualInfo,
    TwoWayFEAbsorber, TwoWaySolveInfo, FEPlan, DofInfo, FixedEffect, Interaction, fe, interaction,
    CategoricalRankInfo, RankBackend, available_rank_backends, categorical_rank, categorical_prefix_ranks,
)
from .results import RegressionResult, EstimationState, FixedEffectEstimates, FixedEffectTermEstimate
from .postestimation import LinearCombinationResult, WaldTestResult
from .bootstrap import wild_bootstrap, wild_cluster_bootstrap_ols, parallel_pairs_bootstrap
from .inference.cluster import (
    ClusterDiagnostics, ClusterDimensionDiagnostics, WildClusterTestResult,
    cluster_diagnostics, wild_cluster_test_ols,
)
from .compute.weights import WeightInfo
from .design import Factor, RegressorInteraction, OmitSpec, factor, reg_interaction, omit_column, omit_term, omit_level
from .factorvars import fv
from .collinearity import OmittedVariableWarning
from .models.linear_iv.stock_yogo import stock_yogo_critical_values
from .sessions import OLSHDFESession, IVHDFESession, OLSSpec, IVSpec

__all__ = [
    "olshdfe", "ppmlhdfe", "ivhdfe", "ivppmlhdfe",
    "reghdfe", "ivreghdfe",
    "HDFEAbsorber", "TwoWayFEAbsorber", "TwoWaySolveInfo", "GroupIndividualAbsorber", "GroupIndividualInfo",
    "RegressionResult", "EstimationState", "FixedEffectEstimates", "FixedEffectTermEstimate", "LinearCombinationResult", "WaldTestResult",
    "FixedEffect", "Interaction", "Factor", "RegressorInteraction", "OmitSpec", "fe", "interaction",
    "factor", "reg_interaction", "fv", "omit_column", "omit_term", "omit_level", "HDFEConvergenceError",
    "DofInfo", "WeightInfo", "wild_bootstrap", "wild_cluster_bootstrap_ols", "parallel_pairs_bootstrap", "cluster_diagnostics", "wild_cluster_test_ols", "ClusterDiagnostics", "ClusterDimensionDiagnostics", "WildClusterTestResult",
    "EconHDFEError", "InputError", "DataTypeError", "ShapeError", "MissingDataError", "InvalidWeightError",
    "SpecificationError", "IdentificationError", "UnderidentifiedError", "CollinearityError", "ConvergenceError",
    "NumericalError", "LinearAlgebraError", "DivergenceError", "InferenceError", "BootstrapError", "VariableRole", "PreflightIssue",
    "PreflightReport", "preflight_dataframe",
    "HDFEConfig", "InferenceConfig", "ExecutionConfig", "ExecutionContext", "PublicationResult", "publication_output",
    "OmittedVariableWarning", "FEPlan", "CategoricalRankInfo", "RankBackend", "available_rank_backends",
    "categorical_rank", "categorical_prefix_ranks", "OLSHDFESession", "IVHDFESession", "OLSSpec", "IVSpec", "PPMLHDFE", "PPMLConfig", "PPMLResult", "IVPPMLHDFE", "IVPPMLConfig", "IVPPMLResult", "SPJPanel", "SPJResult", "ivppml_spj", "SPJBootstrapResult", "ivppml_spj_bootstrap", "stock_yogo_critical_values",
]
__version__ = "0.6.5"
