import numpy as np

from findspingroup.core import Molecule
from findspingroup.core.pg_analyzer import cluster_sites


def test_cluster_sites_uses_single_link_connectivity_on_radial_distances():
    mol = Molecule(
        ["A", "A", "A", "A"],
        [
            [0.03, 0.0, 0.0],
            [0.045, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.11, 0.0, 0.0],
        ],
    )

    origin_site, clustered_sites = cluster_sites(mol, 0.02)

    assert origin_site is None
    cluster_sizes = sorted(len(sites) for sites in clustered_sites.values())
    assert cluster_sizes == [1, 3]

    chained_cluster = max(clustered_sites.values(), key=len)
    chained_radii = sorted(float(np.linalg.norm(site.coords)) for site in chained_cluster)
    assert chained_radii == [0.03, 0.045, 0.06]


def test_cluster_sites_returns_origin_index_when_requested():
    mol = Molecule(
        ["origin", "A", "A"],
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ],
    )

    origin_index, clustered_sites = cluster_sites(mol, 0.02, give_only_index=True)

    assert origin_index == 0
    assert sorted(len(indices) for indices in clustered_sites.values()) == [1, 1]
