from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from competition_api import Sample

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "lib"))

from prep_fused import N_STATS, prep_fast

ONNX = HERE / "cnn_fast_int8.onnx"
EARLY = HERE / "early_linear.npz"

class Model:
    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(ONNX), so, providers=["CPUExecutionProvider"])
        self._xn = np.empty((4, 256), dtype=np.float32)
        self._st = np.empty(N_STATS, dtype=np.float32)
        self._out = np.empty((1, 8), dtype=np.float32)
        self._out0 = self._out[0]
        self._x1 = self._xn[None]
        self._s1 = self._st[None]
        z = np.load(EARLY)
        self._ee = np.ascontiguousarray(z["w"], dtype=np.float32)
        self._ee_b = float(z["b"])
        thr = float(z["threshold"])
        self._ee_thr_z = math.log(thr / (1.0 - thr))
        ov_x = ort.OrtValue.ortvalue_from_numpy(self._x1)
        ov_s = ort.OrtValue.ortvalue_from_numpy(self._s1)
        ov_o = ort.OrtValue.ortvalue_from_numpy(self._out)
        self._ov = (ov_x, ov_s, ov_o)
        self._binding = self._sess.io_binding()
        self._binding.bind_ortvalue_input("xn", ov_x)
        self._binding.bind_ortvalue_input("stats", ov_s)
        self._binding.bind_ortvalue_output("logits", ov_o)
        warm = np.zeros((4, 256), dtype=np.float32)
        warm[:, 17:23] = np.nan
        warm[1, 100:120] = np.nan
        warm.setflags(write=False)
        for sr in (1.0, 2.0):
            for _ in range(3):
                prep_fast(warm, sr, self._xn, self._st)
                self._sess.run_with_iobinding(self._binding)
                np.argmax(self._out[0])
        self._backend = "onnx-int8"

    def predict(self, x: Sample) -> int:
        prep = prep_fast
        sess = self._sess
        binding = self._binding
        xn = self._xn
        st = self._st
        out0 = self._out0
        ee = self._ee
        ee_b = self._ee_b
        ee_thr_z = self._ee_thr_z
        prep(x.series, float(x.meta["sample_rate"]), xn, st)
        if float(np.dot(st, ee)) + ee_b >= ee_thr_z:
            return 1
        sess.run_with_iobinding(binding)
        return int(out0.argmax())

    def logits(self, x: Sample) -> np.ndarray:
        prep = prep_fast
        sess = self._sess
        binding = self._binding
        xn = self._xn
        st = self._st
        out0 = self._out0
        prep(x.series, float(x.meta["sample_rate"]), xn, st)
        sess.run_with_iobinding(binding)
        return out0.copy()
