"""_summary_
    Module that performs coding and encoding of surveillance cameras video files
"""
from importlib.metadata import requires


# requires moviepy and fbpca
# !pip install fbpca
# !pip install moviepy

from .encoder import Encoder
from .decoder import Decoder
