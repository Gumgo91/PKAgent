import pytest

from pkagent.spec import SpecError, compile_model, describe, normalize


def test_defaults_from_suggestions_and_compile():
    n = normalize({'structure': {'compartments': 1, 'absorption': 'first_order', 'lag': True},
                   'iiv': {'CL': .1, 'V': .1}}, suggestions={'CL': 3., 'V': 30., 'Ka': 1.5})
    assert n['parameters']['CL']['value'] == 3. and n['parameters']['Ka']['value'] == 1.5
    assert set(n['defaulted']) == {'CL', 'V', 'Ka', 'ALAG'}
    compile_model(n)
    assert '1-cmt, first_order, lag' in describe(n)


def test_two_compartment_bounds_and_covariate():
    n = normalize({'structure': {'compartments': 2, 'absorption': 'iv'},
                   'parameters': {'CL': 4, 'V1': {'value': 8, 'upper': 10}},
                   'iiv': {'CL': .1, 'V1': .1}, 'iiv_blocks': [['CL', 'V1']],
                   'covariates': [{'parameter': 'CL', 'covariate': 'CLCR', 'form': 'power', 'center': 58}],
                   'residual': {'proportional': .1, 'additive': .1}})
    assert n['parameters']['V1']['upper'] == 10
    assert n['covariates'][0]['coefficient']['value'] == .5
    compile_model(n)


def test_pkpd_outputs_need_residual_per_output():
    with pytest.raises(SpecError, match='residual model per output'):
        normalize({'structure': {'type': 'indirect_response', 'response_type': 1, 'absorption': 'first_order'},
                   'residual': {'proportional': .1}})
    n = normalize({'structure': {'type': 'indirect_response', 'response_type': 1, 'absorption': 'first_order'},
                   'residual': {'CP': {'proportional': .1}, 'R': {'additive': 5}}})
    assert n['outputs'] == ['CP', 'R']
    compile_model(n)


@pytest.mark.parametrize('spec, message', [
    ({'structure': {'compartments': 1}, 'parameters': {'CLX': 1}}, 'not in this structure'),
    ({'structure': {'compartments': 1}, 'iiv': {'CL': .1},
      'covariates': [{'parameter': 'CL', 'covariate': 'SEX', 'form': 'categorical'}]}, 'needs "level"'),
    ({'structure': {'compartments': 4}}, 'compartments'),
    ({'structure': {'compartments': 1}, 'parameters': {'CL': -1}}, 'positive'),
    ({'structure': {'compartments': 1}, 'unknown_key': 1}, 'unknown specification keys'),
    ({'structure': {'compartments': 1}, 'iov': {'CL': .05}}, 'occasion_column'),
])
def test_errors_are_actionable(spec, message):
    with pytest.raises(SpecError, match=message):
        normalize(spec)
