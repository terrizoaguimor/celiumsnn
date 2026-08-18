# SPDX-License-Identifier: AGPL-3.0-or-later
"""celiumsnn — SNN model architecture under the CeliumNeUR constraint set.

P1 scope: integer LIF dynamics (celiumsnn.lif), bit-exact against
celiumneur/golden under the C1-C5 contract of P0-SEMANTICS.md §4.
"""

from celiumsnn.lif import (
    IntLIF,
    VMEM_MAX,
    VMEM_MIN,
    WEIGHT_MAX,
    WEIGHT_MIN,
    ceiling_leak_amount,
    saturate_vmem,
)
from celiumsnn.gates import GatedBlockSparseSynapse, HardConcreteGate, topology_hash
from celiumsnn.lif_diff import DiffLIF
from celiumsnn.quant import QUANTIZERS, quantize_int8, quantize_ternary, quantize_ternary_over
from celiumsnn.surrogate import SHAPES, spike, surrogate_kernel
from celiumsnn.synapse import BlockSparseSynapse, EdgeListSynapse

__all__ = [
    "IntLIF",
    "DiffLIF",
    "EdgeListSynapse",
    "BlockSparseSynapse",
    "GatedBlockSparseSynapse",
    "HardConcreteGate",
    "topology_hash",
    "VMEM_MAX",
    "VMEM_MIN",
    "WEIGHT_MAX",
    "WEIGHT_MIN",
    "ceiling_leak_amount",
    "saturate_vmem",
    "QUANTIZERS",
    "quantize_int8",
    "quantize_ternary",
    "SHAPES",
    "spike",
    "surrogate_kernel",
]
