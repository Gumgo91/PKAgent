"""Functional groups of the 14 PKAgent tools, as shown in Figure 1 and Table S1.

The grouping is descriptive (it is not part of the agent); the assertion keeps it in step with src/pkagent/tools.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pkagent.tools import TOOLS          # noqa: E402

TOOL_GROUPS = {
    'Data': ['describe_data', 'plot_data', 'run_nca', 'add_data_column'],
    'Models': ['fit_models', 'list_models', 'get_model', 'compare_models'],
    'Diagnostics': ['view_plots', 'screen_covariates', 'run_vpc'],
    'Covariates and uncertainty': ['covariate_search', 'resample_uncertainty'],
    'Report': ['finalize_model'],
}


def tool_names():
    return [t['function']['name'] if 'function' in t else t['name'] for t in TOOLS]


_grouped = [name for names in TOOL_GROUPS.values() for name in names]
assert sorted(_grouped) == sorted(tool_names()) and len(_grouped) == len(set(_grouped)), \
    'TOOL_GROUPS does not match the tools in src/pkagent/tools.py'

GROUP_OF = {name: group for group, names in TOOL_GROUPS.items() for name in names}
N_TOOLS = len(_grouped)
