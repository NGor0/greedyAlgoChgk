import os
import sys

module_names = ['ratings-collector']


def enable_import():
  for module_name in module_names:
    sys.path.append(os.path.dirname(os.path.abspath(module_name)))
