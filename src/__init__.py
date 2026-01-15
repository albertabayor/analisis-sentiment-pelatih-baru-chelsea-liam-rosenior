"""
Sentiment Analysis Project - Chelsea Liam Rosenior Appointment

Main package for utility modules.
"""

__version__ = "1.0.0"
__author__ = "Project Team"

from . import utils
from . import preprocessing
from . import feature_engineering
from . import models

__all__ = [
    'utils',
    'preprocessing',
    'feature_engineering',
    'models'
]
