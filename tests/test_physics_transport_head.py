from __future__ import annotations

import torch

from pepp_graph_spib.models.heads.physics_transport_head import PhysicsTransportHead


def test_physics_transport_head_constraints():
    head = PhysicsTransportHead(system_repr_dim=30, condition_dim=18)
    out = head(torch.randn(5, 30), torch.randn(5, 18))
    assert torch.all(out["D_local"] > 0)
    assert torch.all(out["tau_wall"] > 0)
    assert torch.all(out["tau_move"] > 0)
    for key in [
        "P_entry",
        "C_axis",
        "P_access",
        "wall_residence_fraction",
        "transport_score",
    ]:
        assert torch.all((out[key] >= 0) & (out[key] <= 1))
    assert torch.all(out["D_eff"] > 0)
    assert torch.allclose(out["D_eff"], out["D_local"] * out["transport_score"] + head.eps, atol=1.0e-6)
