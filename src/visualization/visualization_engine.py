"""
This module is kept for backwards compatibility.
Please use the modularized version of visualization engine in the following modules:
- src.visualization.base_visualization: Main VisualizationEngine class
- src.visualization.chart_generators: Chart generator classes
- src.visualization.data_validators: Data validation utilities
- src.visualization.utility_functions: Utility functions
- src.visualization.font_manager: Font management utilities
"""

import logging
import warnings
import platform

# Import from modularized modules
from src.visualization.base_visualization import VisualizationEngine
from src.visualization.utility_functions import configure_korean_font, reset_font_cache

# Show deprecation warning
warnings.warn(
    "The monolithic visualization_engine module is deprecated. "
    "Please use the modularized version in src.visualization.*",
    DeprecationWarning, 
    stacklevel=2
)

# Configure Korean fonts on import
configure_korean_font()

# For backwards compatibility
logger = logging.getLogger(__name__) 