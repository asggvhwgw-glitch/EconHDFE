"""Structured covariance computation; no AI services or runtime model calls."""

from .hac import hac_meat, cov_hac
from .cluster import cluster_meat

__all__ = ["hac_meat", "cov_hac", "cluster_meat"]
__version__ = "0.1.0"
