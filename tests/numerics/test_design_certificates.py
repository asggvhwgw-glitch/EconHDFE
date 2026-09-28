import numpy as np
from scipy.linalg import qr

from econhdfe.compute.design_plan import analyze_design_structure
from econhdfe.compute.linalg import independent_columns


def _legacy_independent_columns(X, tol):
    _, R, piv = qr(np.asarray(X, dtype=np.float64), mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    scale = diag[0] if diag.size else 0.0
    rank = int(np.sum(diag > tol * max(scale, 1.0)))
    keep = np.sort(np.asarray(piv[:rank], dtype=np.int64))
    drop = np.setdiff1d(np.arange(X.shape[1]), keep, assume_unique=True)
    return keep, drop


def test_exact_block_certificate_for_disconnected_fe_design():
    rng = np.random.default_rng(1201)
    blocks, rows_per, width = 4, 70, 3
    n = blocks * rows_per
    X = np.zeros((n, blocks * width))
    year = np.repeat(np.arange(blocks), rows_per)
    # FE levels are deliberately unique to each block.
    firm = np.repeat(np.arange(blocks * 14), 5)
    for b in range(blocks):
        rows = year == b
        X[rows, b * width:(b + 1) * width] = rng.normal(size=(rows.sum(), width))

    p = analyze_design_structure(X, groups=[firm, year])
    assert p.certified_block_separable
    assert p.block_count == blocks
    assert p.effective_max_width == width
    assert all(b.nrows == rows_per for b in p.blocks)
    assert p.block_dense_bytes == n * width * 8
    assert np.isclose(p.block_dense_savings_fraction, 1 - 1 / blocks)
    assert np.all(p.row_blocks >= 0)


def test_cross_block_fe_level_correctly_destroys_block_certificate():
    rng = np.random.default_rng(1202)
    n = 240
    block = np.repeat([0, 1], n // 2)
    X = np.zeros((n, 4))
    X[block == 0, :2] = rng.normal(size=(n // 2, 2))
    X[block == 1, 2:] = rng.normal(size=(n // 2, 2))
    year = block.copy()
    # One firm level spans both years; the FE topology couples both X blocks.
    firm = np.arange(n) // 4
    firm[n // 2:n // 2 + 2] = firm[n // 2 - 2]

    p = analyze_design_structure(X, groups=[firm, year])
    assert not p.certified_block_separable
    assert p.block_count == 1
    assert p.reason == "single_connected_component"


def test_zero_x_fe_bridge_is_not_missed():
    # col0 -- fe(a) -- zero-X row -- fe(y) -- col1.  A support-only detector
    # would falsely split this design; the FE incidence graph must not.
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    g1 = np.array([0, 0, 1], dtype=np.int32)
    g2 = np.array([0, 1, 1], dtype=np.int32)
    p = analyze_design_structure(X, groups=[g1, g2])
    assert not p.certified_block_separable
    assert p.block_count == 1


def test_no_fe_block_structure_uses_row_coactivity():
    X = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 6.0],
        [0.0, 0.0, 7.0, 8.0],
    ])
    p = analyze_design_structure(X)
    assert p.certified_block_separable
    assert p.block_count == 2
    assert tuple(len(b.columns) for b in p.blocks) == (2, 2)


def test_clear_full_rank_fast_path_matches_legacy_qr():
    rng = np.random.default_rng(1203)
    X = rng.normal(size=(5000, 20))
    got = independent_columns(X, tol=1e-12)
    expected = _legacy_independent_columns(X, 1e-12)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_rank_deficient_and_near_collinear_cases_fall_back_with_parity():
    rng = np.random.default_rng(1204)
    x = rng.normal(size=(3000, 5))
    exact = np.column_stack([x, x[:, 0] + 2.0 * x[:, 1]])
    near = np.column_stack([x, x[:, 0] + 2.0 * x[:, 1] + 1e-13 * rng.normal(size=len(x))])
    for X in (exact, near):
        got = independent_columns(X, tol=1e-10)
        expected = _legacy_independent_columns(X, 1e-10)
        assert np.array_equal(got[0], expected[0])
        assert np.array_equal(got[1], expected[1])


def test_randomized_small_graph_matches_literal_reference_connectivity():
    rng = np.random.default_rng(1205)
    for _ in range(40):
        n = int(rng.integers(8, 35))
        k = int(rng.integers(1, 8))
        G = int(rng.integers(0, 4))
        X = rng.normal(size=(n, k))
        X[rng.random((n, k)) < 0.68] = 0.0
        groups = [rng.integers(0, int(rng.integers(2, 8)), size=n, dtype=np.int32) for _ in range(G)]
        p = analyze_design_structure(X, groups=groups)

        # Literal observation/column/FE-node union reference for small cases.
        level_counts = [int(g.max()) + 1 for g in groups]
        offsets = np.cumsum([0] + level_counts[:-1]).tolist()
        col0 = sum(level_counts)
        m = col0 + k
        parent = list(range(m))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a
        for i in range(n):
            nodes = [offsets[g] + int(groups[g][i]) for g in range(G)]
            nodes += [col0 + j for j in range(k) if X[i, j] != 0.0]
            if nodes:
                for node in nodes[1:]:
                    union(nodes[0], node)
        roots = [find(col0 + j) for j in range(k)]
        ref_partition = {
            frozenset(j for j, root in enumerate(roots) if root == r)
            for r in set(roots)
        }
        got_partition = {frozenset(b.columns) for b in p.blocks}
        assert got_partition == ref_partition
