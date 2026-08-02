# Package initializer for app module
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_app_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
