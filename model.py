from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from competition_api import Sample

HERE = Path(__file__).parent
for p in (HERE / "lib", HERE.parent / "lib"):
    if p.exists():
        sys.path.insert(0, str(p))

from prep_fused import N_STATS, prep_fast

ONNX_INT8 = HERE / "cnn_fast_int8.onnx"
ONNX_FP32 = HERE / "cnn_fast.onnx"
WEIGHTS = HERE / "cnn_final.pt"
LAG = 8

class Model:
    def __init__(self):
        self._sess = None
        self._binding = None
        self._torch_net = None
        self._backend = "torch"
        self._xn = np.empty((4, 256), dtype=np.float32)
        self._st = np.empty(N_STATS, dtype=np.float32)
        self._out = np.empty((1, 8), dtype=np.float32)
        self._x1 = self._xn[None]
        self._s1 = self._st[None]

        for tag, path in (("onnx-int8", ONNX_INT8), ("onnx-fp32", ONNX_FP32)):
            if self._sess is not None or not path.exists():
                continue
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sess = ort.InferenceSession(
                    str(path), so, providers=["CPUExecutionProvider"])
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
                self._backend = tag
            except Exception:
                self._sess = None
                self._binding = None

        if self._sess is None:
            import torch
            from common import fill_nan
            from net import Net
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)
            ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            net = Net(lag=ck.get("lag", LAG))
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self._torch_net = net
            self._fill_nan = fill_nan

    def _fill(self, x: Sample) -> None:
        prep_fast(np.asarray(x.series, dtype=np.float32),
                  float(x.meta.get("sample_rate", 1.0)), self._xn, self._st)

    def predict(self, x: Sample) -> int:
        if self._sess is not None:
            self._fill(x)
            self._sess.run_with_iobinding(self._binding)
            return int(np.argmax(self._out[0]))
        import torch
        a = np.array(self._fill_nan(np.asarray(x.series, dtype=np.float32)), dtype=np.float32, copy=True)
        nf = float(np.isnan(x.series).mean())
        e = np.array([[nf, float(x.meta.get("sample_rate", 1.0))]], dtype=np.float32)
        with torch.inference_mode():
            L = self._torch_net(torch.from_numpy(a).unsqueeze(0),
                                torch.from_numpy(e)).numpy()[0]
        return int(np.argmax(L))

    def logits(self, x: Sample) -> np.ndarray:
        if self._sess is not None:
            self._fill(x)
            self._sess.run_with_iobinding(self._binding)
            return self._out[0].copy()
        import torch
        a = np.array(self._fill_nan(np.asarray(x.series, dtype=np.float32)), dtype=np.float32, copy=True)
        nf = float(np.isnan(x.series).mean())
        e = np.array([[nf, float(x.meta.get("sample_rate", 1.0))]], dtype=np.float32)
        with torch.inference_mode():
            return self._torch_net(torch.from_numpy(a).unsqueeze(0),
                                   torch.from_numpy(e)).numpy()[0]
