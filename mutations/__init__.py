from __future__ import annotations

from .bool_mask_rewrite import BoolMaskRewrite
from .broadcast_scalarize import BroadcastScalarize
from .cast_simplify import CastSimplify
from .concat_pad import ConcatPadRewrite
from .constant_rewrite import ConstantRewrite
from .conv_rewrite import ConvRewrite
from .dtype_lowering import DtypeLowering
from .dimension_sweep import DimensionSweep
from .gather_shrink import GatherShrink
from .identity_cleanup import IdentityCleanup
from .initializer_dedup import InitializerDedup
from .initializer_prune import InitializerPrune
from .output_collapse import OutputCollapse
from .scalar_label_rewrite import ScalarLabelRewrite
from .synth_solver import SynthSolver
from .table_shrink import TableShrink
from .where_ablation import WhereAblation
from .where_simplify import WhereSimplify


def default_families():
    return [
        InitializerPrune(),
        InitializerDedup(),
        IdentityCleanup(),
        CastSimplify(),
        ConstantRewrite(),
        WhereSimplify(),
        WhereAblation(),
        OutputCollapse(),
        GatherShrink(),
        TableShrink(),
        DimensionSweep(),
        DtypeLowering(),
        ConcatPadRewrite(),
        BoolMaskRewrite(),
        BroadcastScalarize(),
        ScalarLabelRewrite(),
        SynthSolver(),
        ConvRewrite(),
    ]
