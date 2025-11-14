"""
Python re-implementation of the vessel centerline and stenosis workflow.

Modules in :mod:`centerline_pipeline` expose utilities to load 2-D masks with
annotated endpoints, compute distance-based cost volumes, recover minimal-cost
centerlines, and quantify stenosis metrics along each path.
"""

__all__ = [
    "io",
    "preprocess",
    "distance",
    "path",
    "metrics",
    "visualization",
    "process",
]
