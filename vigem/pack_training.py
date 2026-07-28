"""Lightning wrapper for the independent pack-relative setwise VIGEM model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl

from metrics import MyAccuracy
from structured_retrieval import attach_per_epoch_dual_test_eval
from vigem.pack_model import PackRelativeSetwiseStickerModel
from vigem.training import VigemInstanceResidualPLModel


class PackRelativeSetwisePLModel(VigemInstanceResidualPLModel):
    """Reuse VIGEM training/evaluation bookkeeping with an independent model."""

    def __init__(
        self,
        args,
        residual_bundle_path: str,
        membership_hash: str,
        trace_dir: str = "",
        per_query_dir: str = "",
        defer_residual_load: bool = False,
    ):
        pl.LightningModule.__init__(self)
        self.args = args
        self.args.save_structured_test_outputs = False
        self.model = PackRelativeSetwiseStickerModel(
            args,
            residual_bundle_path=residual_bundle_path,
            expected_membership_hash=membership_hash,
            defer_residual_load=defer_residual_load,
        )
        self.model.prepare_imgs(args)

        self.valtest_acc5 = MyAccuracy()
        self.valtest_acc30 = MyAccuracy()
        self.valtest_acc90 = MyAccuracy()
        self.valtest_acc_r10 = MyAccuracy()
        self.valtest_acc_r20 = MyAccuracy()
        self.valtest_map = MyAccuracy()
        self._eval_max_cand_len = 0
        attach_per_epoch_dual_test_eval(self, args)

        self.id2name: Dict[int, str] = {}
        with open(args.id2name_path, encoding="utf-8") as handle:
            raw_names = json.load(handle)
        for key, value in raw_names.items():
            self.id2name[int(key)] = value

        self._style_proto_acc_ema: Optional[float] = None
        self._style_gate_ema: Optional[float] = None
        self._reset_eval_diagnostics()
        self.vigem_membership_hash = str(membership_hash)
        self.vigem_trace_dir = str(trace_dir or "")
        self.vigem_per_query_dir = str(per_query_dir or "")
        self._vigem_trace_handle = None
        self._vigem_trace_partial: Optional[Path] = None
        self._vigem_trace_final: Optional[Path] = None
        self._vigem_last_debug = None
        self._vigem_query_scores = []
        self._vigem_latency_ms = []
        self._vigem_steps = None
        self._vigem_outputs_written = False
