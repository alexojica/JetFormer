import os
import sys


def _ensure_project_root_on_syspath():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_syspath()


