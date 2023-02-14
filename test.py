import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph
import pytest

from dataset import ase_atoms_to_jraph_graph
from qm9 import load_qm9
from util import _loss


qm9_data = load_qm9("qm9_data")


@pytest.mark.parametrize("mol", qm9_data[:4], ids=["CH4", "NH3", "OH2", "C2H2", "CNH"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
def test_loss(mol, graph, output, res_beta, res_alpha, quadrature, gamma=30):
    graph = ase_atoms_to_jraph_graph(mol)
    loss = _loss(output, graph, res_beta, res_alpha, quadrature, gamma)
