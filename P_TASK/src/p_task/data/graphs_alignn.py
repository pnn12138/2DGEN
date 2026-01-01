"""Graph construction helpers for ALIGNN-style models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
from pymatgen.core import Structure


@dataclass
class GraphData:
    """Lightweight container holding the ALIGNN graph inputs."""

    num_nodes: int
    node_feats: torch.Tensor  # [N, node_dim]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor  # [E, edge_dim]
    line_index: torch.Tensor  # [2, L]
    line_attr: torch.Tensor  # [L, line_dim]


def build_alignn_graph(
    structure: Structure,
    cutoff: float = 5.0,
    max_neighbors: int = 16,
    num_rbf: int = 32,
    num_abf: int = 8,
) -> GraphData:
    """Convert a pymatgen Structure into ALIGNN graph tensors with RBF/ABF features."""

    coords = np.array([s.coords for s in structure], dtype=float)
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=int)

    node_feats = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(-1)

    edges: List[tuple[int, int, float, np.ndarray]] = []
    neighbors_all = structure.get_all_neighbors(
        cutoff, include_index=True, include_image=True
    )
    lattice_matrix = np.array(structure.lattice.matrix, dtype=float)
    for src, neighs in enumerate(neighbors_all):
        neighs = sorted(neighs, key=lambda n: n[1])[:max_neighbors]
        for neigh in neighs:
            dst = neigh.index
            dist = float(neigh[1])
            # Apply periodic image translation to get the shortest connector.
            image = np.array(neigh.image, dtype=float)
            translation = image @ lattice_matrix
            vec = coords[dst] + translation - coords[src]
            edges.append((src, dst, dist, vec))

    if not edges:
        raise ValueError("No neighbors found for structure; consider increasing cutoff.")

    edge_src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    edge_dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_dist = torch.tensor([e[2] for e in edges], dtype=torch.float32).unsqueeze(-1)
    edge_attr = _gaussian_basis(
        edge_dist,
        centers=torch.linspace(0, cutoff, steps=num_rbf, dtype=torch.float32),
        width=_basis_width(cutoff, num_rbf),
    )
    edge_vecs = [e[3] for e in edges]

    # Build line-graph connections for all edge pairs that share the same source node.
    line_src: List[int] = []
    line_dst: List[int] = []
    line_attr: List[float] = []
    edges_by_node: List[List[int]] = [[] for _ in range(len(structure))]
    for idx, (src, dst, _, _) in enumerate(edges):
        edges_by_node[src].append(idx)
        edges_by_node[dst].append(idx)

    for node_edge_ids in edges_by_node:
        for i, e_i in enumerate(node_edge_ids):
            for e_j in node_edge_ids[i + 1 :]:
                vec_i = edge_vecs[e_i]
                vec_j = edge_vecs[e_j]
                cos_theta = _safe_cosine(vec_i, vec_j)
                # Add both directions so the line graph stays bidirectional.
                line_src.extend([e_i, e_j])
                line_dst.extend([e_j, e_i])
                line_attr.extend([cos_theta, cos_theta])

    if not line_src:
        # Avoid empty line-graph (rare for tiny structures); fall back to zeros.
        line_src = [0]
        line_dst = [0]
        line_attr = [1.0]

    line_index = torch.stack(
        [torch.tensor(line_src, dtype=torch.long), torch.tensor(line_dst, dtype=torch.long)],
        dim=0,
    )
    line_attr_tensor = torch.tensor(line_attr, dtype=torch.float32).unsqueeze(-1)
    line_attr = _gaussian_basis(
        line_attr_tensor,
        centers=torch.linspace(-1.0, 1.0, steps=num_abf, dtype=torch.float32),
        width=_basis_width(2.0, num_abf),
    )

    return GraphData(
        num_nodes=len(structure),
        node_feats=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_index=line_index,
        line_attr=line_attr,
    )


def collate_graphs(graphs: Sequence[GraphData]) -> dict:
    """Batch a list of GraphData objects into a single tensor dict."""

    node_offset = 0
    edge_offset = 0
    node_feats = []
    edge_index = []
    edge_attr = []
    line_index = []
    line_attr = []
    node_batch = []

    for batch_id, g in enumerate(graphs):
        node_feats.append(g.node_feats)
        edge_index.append(g.edge_index + node_offset)
        edge_attr.append(g.edge_attr)
        line_index.append(g.line_index + edge_offset)
        line_attr.append(g.line_attr)
        node_batch.append(torch.full((g.num_nodes,), batch_id, dtype=torch.long))

        node_offset += g.num_nodes
        edge_offset += g.edge_index.shape[1]

    return {
        "node_feats": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_attr": torch.cat(edge_attr, dim=0),
        "line_index": torch.cat(line_index, dim=1),
        "line_attr": torch.cat(line_attr, dim=0),
        "node_batch": torch.cat(node_batch, dim=0),
    }


def _safe_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0
    return float(np.dot(v1, v2) / denom)


def _basis_width(span: float, num: int) -> float:
    """Heuristic width so neighboring centers overlap."""
    if num <= 1:
        return span if span > 0 else 1.0
    return span / (num - 1) * 0.5


def _gaussian_basis(values: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    """Expand scalar values with Gaussian radial/angle basis."""
    width = max(width, 1e-6)
    diff = values - centers
    return torch.exp(-0.5 * (diff / width) ** 2)
