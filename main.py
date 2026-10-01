from __future__ import annotations

# ----------------------- USER-EDITABLE SETTINGS -----------------------
MODEL_NAME = "./models/qwen3_4b"
# Alternative: "./models/gemma4b"
MODEL_REVISION = "main"  # Resolved to an immutable revision and saved at runtime.
OUTPUT_DIR = "./acl_results"
DEVICE = "cuda:0"
MODEL_DTYPE = "bfloat16"
CONTEXT_LENGTHS = (4096, 8192, 16384)
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
LEAF_SIZE = 64
INITIAL_BLOCK_SIZE = 256
QUEST_PAGE_SIZE = 16
STAGE_CAPS = (1, 2, 4)
QUEST_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
MAGICPIG_SETTINGS = ((9, 128), (9, 256), (9, 512),
                     (10, 128), (10, 256), (10, 512))
RANDOM_SEEDS = (17, 31, 47)
DATA_SEED = 13
MEAN_TV_LIMIT = 1e-3
P99_TV_LIMIT = 1e-2
RULER_DIRECTORY = ""  # Optional directory of already prepared official JSONL data.
RULER_REVISION = "main"  # Resolved to a commit before optional preparation.
LONG_BENCH_REVISION = "main"
FINEWEB_REVISION = "main"
# ---------------------------------------------------------------------

import argparse
import collections
import contextlib
import copy
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import io
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import string
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
_USED_HF_STREAMING = False

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("Install numpy and the appropriate PyTorch build first.") from exc

LOG = logging.getLogger("rea")
MIB = 2 ** 20
CERTIFICATE_STATUS = "numerically_validated_not_formally_verified"
ALIASES = {"qwen": "Qwen/Qwen3-4B-Instruct-2507",
           "llama": "meta-llama/Llama-3.2-3B-Instruct"}
SUPPORTED_TYPES = {"qwen3", "llama"}
LONG_TASKS = ("qasper", "hotpotqa", "2wikimqa", "multi_news")
RULER_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multiquery", "vt")


@dataclasses.dataclass
class Config:
    models: tuple[str, ...] = (MODEL_NAME,)
    revision: str = MODEL_REVISION
    out: str = OUTPUT_DIR
    device: str = DEVICE
    dtype: str = MODEL_DTYPE
    suite: str = "smoke"
    datasets: tuple[str, ...] = ("fineweb",)
    contexts: tuple[int, ...] = CONTEXT_LENGTHS
    temperature: float = TEMPERATURE
    top_k: int = TOP_K
    top_p: float = TOP_P
    leaf: int = LEAF_SIZE
    initial: int = INITIAL_BLOCK_SIZE
    page: int = QUEST_PAGE_SIZE
    stages: tuple[int, ...] = STAGE_CAPS
    quest_fractions: tuple[float, ...] = QUEST_FRACTIONS
    magic_settings: tuple[tuple[int, int], ...] = MAGICPIG_SETTINGS
    seeds: tuple[int, ...] = RANDOM_SEEDS
    n_eval: int = 128
    n_cal: int = 32
    max_long_examples: int = 0
    prefixes: int = 64
    teacher_tokens: int = 128
    warmups: int = 10
    repeats: int = 30
    bootstrap: int = 2000
    mean_tv: float = MEAN_TV_LIMIT
    p99_tv: float = P99_TV_LIMIT
    law_tol: float = 1e-9
    logit_tol: float = 1e-5
    interval_chunk: int = 512
    data_seed: int = DATA_SEED
    generate_tasks: bool = True
    eval_grid: bool = True
    prepare_ruler: bool = False
    ruler_dir: str = RULER_DIRECTORY
    ruler_revision: str = RULER_REVISION
    fineweb_revision: str = FINEWEB_REVISION
    longbench_revision: str = LONG_BENCH_REVISION
    profile_prompts: int = 1
    profile_prefixes: int = 1
    ncu: str = "ncu"
    allow_other_transformers: bool = False
    overwrite: bool = False
    # Used by private counter workers, not a user-specified performance result.
    counter_worker: bool = False


@dataclasses.dataclass(frozen=True)
class Policy:
    kind: str
    parameter: float
    temperature: float

    @property
    def key(self) -> str:
        return f"{self.kind}_{self.parameter:g}_t{self.temperature:g}"

    @property
    def label(self) -> str:
        return f"Top-{int(self.parameter)}" if self.kind == "topk" else f"Nucleus {self.parameter:g}"


@dataclasses.dataclass(frozen=True)
class Method:
    kind: str
    degree: int = 1
    stages: int = 2
    fraction: float = 1.0
    bits: int = 9
    tables: int = 128
    support_first: bool = False
    dense_endpoint: bool = False

    @property
    def key(self) -> str:
        if self.kind == "dense":
            return "dense"
        if self.dense_endpoint:
            return self.kind + "_dense"
        if self.kind == "quest":
            return f"quest_{self.fraction:g}"
        if self.kind == "magicpig":
            return f"magicpig_K{self.bits}_L{self.tables}"
        mode = "support" if self.support_first else "progressive"
        return f"{mode}_d{self.degree}_s{self.stages}"

    @property
    def label(self) -> str:
        names = {"dense": "Dense", "quest": "Quest (PyTorch)",
                 "magicpig": "MagicPIG (GPU, PyTorch)", "progressive": "Progressive (numeric)"}
        if self.support_first:
            return "Support first (numeric)"
        return names[self.kind]

    @property
    def is_dense(self) -> bool:
        return self.kind == "dense" or self.dense_endpoint or (self.kind == "quest" and self.fraction >= 1)


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def clean_json(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return clean_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Malformed JSON at {path}:{i}. No record was skipped.") from exc
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json(row), ensure_ascii=False) + "\n")
        f.flush()


def sync(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, math.inf))


def gamma(n: int, dtype: torch.dtype = torch.float64) -> float:
    u = torch.finfo(dtype).eps / 2
    return n * u / (1 - n * u) if n * u < 1 else math.inf


@dataclasses.dataclass
class Box:
    lo: torch.Tensor
    hi: torch.Tensor

    @staticmethod
    def point(x: torch.Tensor) -> "Box":
        y = x.to(torch.float64)
        return Box(y, y)

    @staticmethod
    def bounds(lo: torch.Tensor, hi: torch.Tensor) -> "Box":
        return Box(lo.to(torch.float64), hi.to(torch.float64))

    @property
    def mid(self) -> torch.Tensor:
        return self.lo * 0.5 + self.hi * 0.5

    @property
    def rad(self) -> torch.Tensor:
        return (self.hi - self.lo) * 0.5

    def __add__(self, other: "Box" | torch.Tensor | float) -> "Box":
        b = other if isinstance(other, Box) else Box.point(torch.as_tensor(other, device=self.lo.device))
        return Box(down(self.lo + b.lo), up(self.hi + b.hi))

    def __mul__(self, other: "Box" | torch.Tensor | float) -> "Box":
        b = other if isinstance(other, Box) else Box.point(torch.as_tensor(other, device=self.lo.device))
        products = torch.stack([self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi])
        return Box(down(products.amin(0)), up(products.amax(0)))

    def scale(self, a: torch.Tensor | float) -> "Box":
        return self * a

    def divide_positive(self, denominator: "Box") -> "Box":
        if bool(torch.any(denominator.lo <= 0)):
            raise ValueError("A positive denominator is required.")
        return self * Box(down(1 / denominator.hi), up(1 / denominator.lo))

    def stored32(self) -> "Box":
        # Both endpoints are binary32, rounded one representable step outward.
        return Box(down(self.lo.float()), up(self.hi.float()))

    def as64(self) -> "Box":
        return Box.bounds(self.lo, self.hi)

    def inflate(self, e: torch.Tensor | float) -> "Box":
        return Box(down(self.lo-e), up(self.hi+e))

    def intersect(self, other: "Box") -> "Box":
        lo, hi = torch.maximum(self.lo, other.lo), torch.minimum(self.hi, other.hi)
        if bool(torch.any(lo > hi)):
            raise ArithmeticError("Empty interval intersection. The run is invalid.")
        return Box(lo, hi)


def model_round(box: Box, dtype: torch.dtype) -> Box:
    if dtype == torch.float64:
        return box.inflate(8 * torch.finfo(torch.float64).eps * (1 + box.mid.abs() + box.rad))
    unit = torch.finfo(dtype).eps / 2
    # This is an explicit numerical allowance, not a verified CUDA error theorem.
    return box.inflate((unit / (1 - unit)) * torch.maximum(box.lo.abs(), box.hi.abs())
                       + torch.finfo(dtype).tiny)


def sum_box(x: torch.Tensor, axis: int) -> Box:
    xd = x.double()
    val = xd.sum(axis)
    err = gamma(x.shape[axis] + 2) * xd.abs().sum(axis)
    return Box(down(val-err), up(val+err)).stored32()


def linear_box(x: Box, layer: torch.nn.Linear, output_dtype: torch.dtype,
               chunk: int = 512) -> Box:
    """Row-chunked interval matrix product without retaining fp64 model copies."""
    m, r = x.mid, x.rad
    output_lo, output_hi = [], []
    for start in range(0, layer.weight.shape[0], chunk):
        w = layer.weight[start:start+chunk].double()
        center = F.linear(m, w)
        radius = F.linear(r, w.abs())
        absolute_sum = F.linear(m.abs() + r, w.abs())
        # fp32-accumulated BF16/FP32 products + binary64 interval arithmetic.
        err = (gamma(w.shape[-1]+2, torch.float32) + gamma(w.shape[-1]+4)) * absolute_sum
        if layer.bias is not None:
            b = layer.bias[start:start+chunk].double()
            center = center + b
            err = err + torch.finfo(torch.float32).eps * b.abs()
        out = model_round(Box(down(center-radius-err), up(center+radius+err)), output_dtype)
        output_lo.append(out.lo)
        output_hi.append(out.hi)
    return Box(torch.cat(output_lo, -1), torch.cat(output_hi, -1))


def rms_box(x: Box, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> Box:
    l, u = x.lo, x.hi
    square_lo = torch.where((l <= 0) & (u >= 0), 0, torch.minimum(l*l, u*u))
    square_hi = torch.maximum(l*l, u*u)
    count = l.shape[-1]
    err = gamma(count+4, torch.float32)
    dl = down(torch.sqrt(torch.clamp((1-err)*square_lo.mean(-1, keepdim=True) + eps, min=eps)))
    du = up(torch.sqrt((1+err)*square_hi.mean(-1, keepdim=True) + eps))
    reciprocal = Box(down(1/du), up(1/dl)).inflate(
        8 * torch.finfo(torch.float32).eps / dl)
    # Native RMSNorm casts normalized activations to the model dtype before scale.
    normalized = model_round(x * reciprocal, dtype)
    return model_round(normalized * weight.double(), dtype)


def silu_box(x: Box, dtype: torch.dtype) -> Box:
    # Endpoint product is conservative, including SiLU's negative turning point.
    lo = torch.sigmoid(x.lo)
    hi = torch.sigmoid(x.hi)
    eps = 8 * torch.finfo(torch.float32).eps
    s = Box((lo-eps).clamp(0, 1), (hi+eps).clamp(0, 1))
    return model_round(x * s, dtype)


def policy_prob(logits: torch.Tensor, policy: Policy) -> torch.Tensor:
    z = logits.flatten().double() / policy.temperature
    if not bool(torch.isfinite(z).all()):
        raise ArithmeticError("Dense or approximate logits contain non-finite values.")
    order = torch.argsort(z, descending=True, stable=True)  # token-index ties
    p = torch.zeros_like(z)
    if policy.kind == "topk":
        ids = order[:min(int(policy.parameter), z.numel())]
    elif policy.kind == "nucleus":
        full = torch.softmax(z[order], 0)
        # The shortest prefix whose cumulative mass reaches the cutoff.
        j = int(torch.searchsorted(full.cumsum(0), torch.tensor(policy.parameter,
                    device=z.device, dtype=z.dtype), right=False).item()) + 1
        ids = order[:min(j, z.numel())]
    elif policy.kind == "full":
        ids = order
    else:
        raise ValueError(f"Unknown decoding policy {policy.kind}")
    p[ids] = torch.softmax(z[ids], 0)
    return p


def tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p.double()-q.double()).abs().sum().item())


def draw_categorical(p: torch.Tensor, rng: np.random.Generator) -> int:
    # CPU random draw is independent of tensor arithmetic and index construction.
    # A binary64 draw is not a computable-real random-bit sampler.
    u = float(rng.random())
    cdf = p.double().cumsum(0)
    total = float(cdf[-1])
    if not (math.isfinite(total) and total > 0):
        raise ArithmeticError("Invalid categorical normalizer")
    i = torch.searchsorted(cdf, torch.tensor(u*total, device=p.device, dtype=p.dtype), right=True)
    return min(int(i.item()), p.numel()-1)


@dataclasses.dataclass
class Certificate:
    lower: torch.Tensor
    support: bool
    core: torch.Tensor


def probability_certificate(box: Box, policy: Policy) -> Certificate:
    """O(M log M) independent-logit certificates. Equality is conservative."""
    l = box.lo.flatten().double() / policy.temperature
    u = box.hi.flatten().double() / policy.temperature
    m = l.numel()
    zeros = torch.zeros_like(l)
    if not bool(torch.isfinite(l).all() and torch.isfinite(u).all()):
        return Certificate(zeros, False, zeros.bool())
    if bool(torch.any(l > u)):
        raise ArithmeticError("Invalid logit interval")
    if bool(torch.equal(l, u)):
        p = policy_prob(l, dataclasses.replace(policy, temperature=1.0))
        return Certificate(p, True, p > 0)
    order = torch.argsort(u, descending=True, stable=True)
    su = u[order]
    # Including equal upper bounds on both sides of the declared tie is safe.
    count = torch.searchsorted(-su, -l, right=True)
    possible_competitors = count - 1  # y itself is necessarily in the prefix.
    shift = u.max()
    eu, el = torch.exp(u-shift), torch.exp(l-shift)
    eu_sorted = eu[order]
    el_sorted = el[order]
    prefix_u = torch.cat([eu.new_zeros(1), eu_sorted.cumsum(0)])
    suffix_l = torch.cat([el_sorted.flip(0).cumsum(0).flip(0), el.new_zeros(1)])
    guard = 16 * torch.finfo(torch.float64).eps * (m + 1)
    all_except_y = (prefix_u[-1]-eu).clamp_min(0) + guard*prefix_u[-1]
    full_lower = el / (el + all_except_y).clamp_min(torch.finfo(torch.float64).tiny)
    if policy.kind == "full":
        core = torch.ones_like(l, dtype=torch.bool)
        lower, complete = full_lower, True
    elif policy.kind == "topk":
        k = min(int(policy.parameter), m)
        core = possible_competitors < k
        rank = torch.empty(m, device=l.device, dtype=torch.long)
        rank[order] = torch.arange(m, device=l.device)
        if k == 1:
            denom_comp = torch.zeros_like(l)
        else:
            denom_comp = torch.where(rank < k-1, prefix_u[k]-eu, prefix_u[k-1])
            denom_comp = denom_comp.clamp_min(0) + guard*prefix_u[min(k, m)]
        lower = torch.where(core, el/(el+denom_comp).clamp_min(torch.finfo(l.dtype).tiny), 0)
        complete = bool(core.sum().item() == k)
    elif policy.kind == "nucleus":
        preceding_u = (prefix_u[count]-eu).clamp_min(0) + guard*prefix_u[-1]
        complement_l = (suffix_l[count]+el) * (1-guard)
        upper_mass = preceding_u / (preceding_u + complement_l).clamp_min(torch.finfo(l.dtype).tiny)
        core = upper_mass < policy.parameter
        lower, complete = torch.where(core, full_lower, 0), False
        # One candidate prefix suffices to produce a valid complete certificate.
        mid = (l+u)*0.5
        proposal = policy_prob(mid, dataclasses.replace(policy, temperature=1.0))
        subset = proposal > 0
        baseline_order = torch.argsort(mid, descending=True, stable=True)
        j = int(subset.sum())
        last = baseline_order[j-1]
        others = subset.clone()
        others[last] = False
        outside = ~subset
        sep = (not bool(outside.any())) or bool(l[subset].min() > u[outside].max())
        last_ok = (j == 1) or bool(l[others].min() > u[last])
        sum_l_in, sum_u_out = el[subset].sum(), eu[outside].sum()
        mass_l = ((1-guard)*sum_l_in) / ((1-guard)*sum_l_in+(1+guard)*sum_u_out).clamp_min(1e-300)
        a, b = eu[others].sum(), el[~others].sum()
        before_u = ((1+guard)*a) / ((1+guard)*a+(1-guard)*b).clamp_min(1e-300)
        complete = sep and last_ok and bool(mass_l >= policy.parameter) and bool(before_u < policy.parameter)
        if complete:
            core = subset
    else:
        raise ValueError(policy.kind)
    if complete:
        sum_u = eu[core].sum()
        competitors = (sum_u-eu).clamp_min(0) + guard*sum_u
        lower = torch.where(core, el/(el+competitors).clamp_min(1e-300), 0)
    lower = down(lower.clamp_min(0)*(1-guard)).clamp_min(0)
    if float(lower.sum()) > 1+1e-12:
        raise ArithmeticError("Certificate mass exceeds one")
    return Certificate(lower, complete, core)


class SourceAccess:
    """Logical original-record requests; alias and GQA-aware distinct addresses."""
    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v = k, v  # [G, N, d]
        self.g, self.n = k.shape[:2]
        self.requests: list[tuple[int, torch.Tensor]] = []
        self.all_read = False
        self.key_requests = 0
        self.value_requests = 0
        self.bytes = 0

    def get(self, g: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.unique(ids.to(device=self.k.device, dtype=torch.long), sorted=True)
        self.key_requests += ids.numel()
        self.value_requests += ids.numel()
        self.bytes += ids.numel() * (self.k.shape[-1]*self.k.element_size()
                                     + self.v.shape[-1]*self.v.element_size())
        self.requests.append((g, ids.detach()))
        return self.k[g].index_select(0, ids), self.v[g].index_select(0, ids)

    def dense(self) -> None:
        self.key_requests += self.g*self.n
        self.value_requests += self.g*self.n
        self.bytes += nbytes(self.k)+nbytes(self.v)
        self.all_read = True

    @property
    def distinct(self) -> int:
        if self.all_read:
            return self.g*self.n
        seen=set()
        for g,ids in self.requests:
            seen.update((g,int(i)) for i in ids.detach().cpu().tolist())
        return len(seen)

    def append_access(self) -> None:
        # Tree/index maintenance reads the currently appended physical record.
        for g in range(self.g):
            ids=torch.tensor([self.n-1],device=self.k.device,dtype=torch.long)
            self.requests.append((g,ids))
        self.key_requests+=self.g;self.value_requests+=self.g
        self.bytes+=self.g*(self.k.shape[-1]*self.k.element_size()+self.v.shape[-1]*self.v.element_size())


@dataclasses.dataclass
class Node:
    start: int
    end: int
    n: int
    kmin: torch.Tensor
    kmax: torch.Tensor
    vmin: torch.Tensor
    vmax: torch.Tensor
    vradius: torch.Tensor
    vsum: Box
    ksum: Optional[Box] = None
    cross: Optional[Box] = None


class SummaryTree:
    """Persistent endpoint summaries shared by physical KV heads.

    Snapshots restore references, never reverse floating-point additions. Appends
    read only the appended original record and update one root-to-leaf path.
    """
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, leaf: int,
                 initial: int, degree: int, max_records: int):
        self.leaf, self.initial, self.degree = leaf, initial, degree
        self.g, self.d = keys.shape[0], keys.shape[-1]
        leaves = max(1, math.ceil(max_records/leaf))
        self.capacity = 1 << (leaves-1).bit_length()
        self.nodes: dict[int, Node] = {}
        self.n = keys.shape[1]
        for j, start in enumerate(range(0, self.n, leaf)):
            end = min(start+leaf, self.n)
            self.nodes[self.capacity+j] = self.make_node(keys[:, start:end], values[:, start:end], start, end)
        for idx in range(self.capacity-1, 0, -1):
            children = [self.nodes[c] for c in (2*idx, 2*idx+1) if c in self.nodes]
            if children:
                self.nodes[idx] = children[0] if len(children) == 1 else self.merge(*children)
        self._undo: Optional[tuple[int, dict[int, Optional[Node]]]] = None

    def make_node(self, k: torch.Tensor, v: torch.Tensor, start: int, end: int) -> Node:
        kd, vd = k.double(), v.double()
        count = k.shape[1]
        cross = None
        if self.degree == 1:
            c = torch.einsum("gnv,gnk->gvk", vd, kd)
            abs_c = torch.einsum("gnv,gnk->gvk", vd.abs(), kd.abs())
            err = gamma(count+3)*abs_c
            cross = Box(down(c-err), up(c+err)).stored32()
        return Node(start, end, count, k.amin(1).float(), k.amax(1).float(),
                    v.amin(1).float(), v.amax(1).float(),
                    up(vd.norm(dim=-1).amax(1)*(1+gamma(v.shape[-1]+4))).float(),
                    sum_box(v, 1), sum_box(k, 1) if self.degree else None, cross)

    def merge(self, a: Node, b: Node) -> Node:
        return Node(min(a.start,b.start), max(a.end,b.end), a.n+b.n,
                    torch.minimum(a.kmin,b.kmin), torch.maximum(a.kmax,b.kmax),
                    torch.minimum(a.vmin,b.vmin), torch.maximum(a.vmax,b.vmax),
                    torch.maximum(a.vradius,b.vradius),
                    (a.vsum.as64()+b.vsum.as64()).stored32(),
                    (a.ksum.as64()+b.ksum.as64()).stored32() if self.degree else None,
                    (a.cross.as64()+b.cross.as64()).stored32() if self.degree else None)

    def snapshot(self) -> tuple[int, dict[int, Node]]:
        return self.n, dict(self.nodes)

    def restore(self, state: tuple[int, dict[int, Node]]) -> None:
        self.n, nodes = state
        self.nodes = dict(nodes)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.n >= self.capacity*self.leaf:
            raise RuntimeError("Tree capacity exhausted; increase max_records.")
        idx = self.capacity + self.n//self.leaf
        new = self.make_node(k[:, None], v[:, None], self.n, self.n+1)
        self.nodes[idx] = self.merge(self.nodes[idx], new) if idx in self.nodes else new
        idx //= 2
        while idx:
            children = [self.nodes[c] for c in (2*idx,2*idx+1) if c in self.nodes]
            self.nodes[idx] = children[0] if len(children)==1 else self.merge(*children)
            idx //= 2
        self.n += 1

    def initial_frontier(self) -> list[int]:
        out: list[int] = []
        def visit(idx: int) -> None:
            if idx not in self.nodes:
                return
            n = self.nodes[idx]
            if n.n <= self.initial or idx >= self.capacity:
                out.append(idx)
            else:
                visit(2*idx)
                visit(2*idx+1)
        visit(1)
        return out

    def storage_bytes(self) -> int:
        seen = set()
        total = 0
        for node in self.nodes.values():
            for value in vars(node).values():
                seq = (value.lo,value.hi) if isinstance(value,Box) else (value,)
                for t in seq:
                    if torch.is_tensor(t) and t.data_ptr() not in seen:
                        seen.add(t.data_ptr()); total += nbytes(t)
        return total


def dot_interval(q: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> Box:
    qp, qn = q.clamp_min(0), q.clamp_max(0)
    l = (qp*lo + qn*hi).sum(-1)
    u = (qp*hi + qn*lo).sum(-1)
    err = gamma(q.shape[-1]+4) * (q.abs()*torch.maximum(lo.abs(),hi.abs())).sum(-1)
    return Box(down(l-err), up(u+err))


class Frontier:
    def __init__(self, tree: SummaryTree, q: torch.Tensor, source: SourceAccess, scaling: float):
        self.tree, self.q, self.source = tree, q.double()*scaling, source
        self.h, self.d = q.shape
        self.repeats = self.h//tree.g
        self.frontiers = {g: tree.initial_frontier()[:] for g in range(tree.g)}
        self.resolved: dict[tuple[int,int], tuple[torch.Tensor,torch.Tensor]] = {}
        self.priorities: dict[tuple[int,int], float] = {}
        self.last_radius = math.inf

    def score_bound(self, g: int, node: Node) -> Box:
        q = self.q[g*self.repeats:(g+1)*self.repeats]
        return dot_interval(q, node.kmin[g].double(), node.kmax[g].double())

    def enclosure(self, output_dtype: torch.dtype) -> Box:
        all_lo, all_hi = [], []
        max_radius = 0.0
        self.priorities = {}
        root = self.tree.nodes[1]
        all_bounds = {g:{idx:self.score_bound(g,self.tree.nodes[idx]) for idx in self.frontiers[g]}
                      for g in range(self.tree.g)}
        common_shift = torch.stack([b.hi.max() for group in all_bounds.values() for b in group.values()]).max()
        for g in range(self.tree.g):
            q = self.q[g*self.repeats:(g+1)*self.repeats]
            bounds = all_bounds[g]
            shift = common_shift.expand(self.repeats)
            A = Box.point(torch.zeros(self.repeats,device=q.device,dtype=torch.float64))
            S = Box.point(torch.zeros(self.repeats,self.source.v.shape[-1],device=q.device,dtype=torch.float64))
            E = torch.zeros_like(shift)
            for idx in self.frontiers[g]:
                node, bound = self.tree.nodes[idx], bounds[idx]
                if (g,idx) in self.resolved:
                    logz, mean = self.resolved[g,idx]
                    a = torch.exp(logz-shift)
                    A = A + Box.point(a).inflate(32*torch.finfo(torch.float64).eps*a)
                    S = S + Box.point(a[:,None]*mean).inflate(64*torch.finfo(torch.float64).eps*(1+mean.abs()))
                    continue
                sl, sh = bound.lo, bound.hi
                width = (sh-sl).clamp_min(0)
                lower_exp = torch.exp(sl-shift)
                upper_exp = torch.exp(sh-shift)
                vsum = Box.bounds(node.vsum.lo[g], node.vsum.hi[g])
                if self.tree.degree == 0:
                    a_box = Box.point(node.n*lower_exp)
                    s_box = vsum * lower_exp[:,None]
                    rem = -torch.expm1(-width)
                else:
                    ks = dot_interval(q,node.ksum.lo[g].double(),node.ksum.hi[g].double())
                    core = ks + node.n*(1-sl)
                    a_box = core * lower_exp
                    # Analytic positivity also bounds this contraction from below.
                    a_box.lo = torch.maximum(a_box.lo, down(node.n*lower_exp))
                    cl, cu = node.cross.lo[g].double(), node.cross.hi[g].double()
                    qp, qn = q.clamp_min(0), q.clamp_max(0)
                    clo = qp@cl.T + qn@cu.T
                    chi = qp@cu.T + qn@cl.T
                    er = gamma(self.d+4) * (q.abs()@torch.maximum(cl.abs(),cu.abs()).T)
                    cq = Box(down(clo-er),up(chi+er))
                    s_box = (vsum*(1-sl[:,None])+cq)*lower_exp[:,None]
                    rem = -torch.expm1(-width)-width*torch.exp(-width)
                e = up(node.n*upper_exp*(rem.clamp_min(0)+64*torch.finfo(torch.float64).eps))
                a_box = a_box.inflate(64*torch.finfo(torch.float64).eps*(1+a_box.mid.abs()))
                A = A+a_box
                S = S+s_box
                E = up(E+e)
                # Compare physical nodes on a common per-group score scale.
                self.priorities[g,idx] = float(e.sum())
            if bool(torch.any(A.lo <= 0)):
                o_box = Box.bounds(root.vmin[g].expand(self.repeats,-1),root.vmax[g].expand(self.repeats,-1))
                radius = float((o_box.hi-o_box.lo).amax())
            else:
                base = S.divide_positive(Box(A.lo[:,None],A.hi[:,None]))
                eta = (E/(A.lo+E)).clamp(0,1)
                max_norm = torch.maximum(base.lo.abs(),base.hi.abs()).norm(dim=-1)
                radius_head = eta*(root.vradius[g].double()+max_norm)
                o_box = base.inflate(radius_head[:,None])
                o_box = o_box.intersect(Box.bounds(root.vmin[g],root.vmax[g]))
                radius = float(radius_head.max())
            # Explicit allowance for the native finite-precision attention path.
            sb = self.score_bound(g,root)
            max_score = torch.maximum(sb.lo.abs(),sb.hi.abs())
            score_error = gamma(self.d+4,torch.float32)*max_score
            prob_error = torch.expm1((2*score_error).clamp_max(40)) + gamma(2*self.tree.n+16,torch.float32)
            value_abs = torch.maximum(root.vmin[g].abs(),root.vmax[g].abs()).double()
            cast_u = torch.finfo(output_dtype).eps/2
            pad = (prob_error[:,None]+2*cast_u+32*torch.finfo(torch.float32).eps)*value_abs
            o_box = o_box.inflate(pad)
            max_radius = max(max_radius,radius+float(pad.max()))
            all_lo.append(o_box.lo); all_hi.append(o_box.hi)
        self.last_radius = max_radius
        return Box(torch.cat(all_lo,0),torch.cat(all_hi,0))

    def refine(self) -> bool:
        available = [(p,self.tree.nodes[idx].start,g,idx) for (g,idx),p in self.priorities.items()
                     if (g,idx) not in self.resolved]
        if not available:
            return False
        _, _, g, idx = min(available,key=lambda t:(-t[0],t[1],t[2]))
        node = self.tree.nodes[idx]
        if idx < self.tree.capacity:
            children = [i for i in (2*idx,2*idx+1) if i in self.tree.nodes]
            place = self.frontiers[g].index(idx)
            self.frontiers[g][place:place+1] = children
        else:
            ids = torch.arange(node.start,node.end,device=self.q.device)
            k,v = self.source.get(g,ids)
            q = self.q[g*self.repeats:(g+1)*self.repeats]
            scores = q@k.double().T
            logz = scores.logsumexp(-1)
            mean = torch.softmax(scores,-1)@v.double()
            self.resolved[g,idx]=(logz,mean)
        return True

    def score_widths(self) -> tuple[list[float],list[float]]:
        # This routine is diagnostic only and is never called by the sampler.
        observed, certified = [], []
        for g in range(self.tree.g):
            q = self.q[g*self.repeats:(g+1)*self.repeats]
            for idx in self.tree.initial_frontier():
                node = self.tree.nodes[idx]
                scores = q@self.source.k[g,node.start:node.end].double().T
                b = self.score_bound(g,node)
                observed.extend((scores.amax(-1)-scores.amin(-1)).cpu().tolist())
                certified.extend((b.hi-b.lo).cpu().tolist())
        return observed, certified


class PageIndex:
    def __init__(self, keys: torch.Tensor, page: int):
        self.page, self.n = page, keys.shape[1]
        self.lo = [keys[:,s:min(s+page,self.n)].amin(1).float() for s in range(0,self.n,page)]
        self.hi = [keys[:,s:min(s+page,self.n)].amax(1).float() for s in range(0,self.n,page)]

    def snapshot(self) -> tuple:
        return self.n,self.lo[:],self.hi[:]

    def restore(self,state:tuple) -> None:
        self.n,self.lo,self.hi=state[0],state[1][:],state[2][:]

    def append(self,k:torch.Tensor,v:torch.Tensor) -> None:
        if self.n % self.page == 0:
            self.lo.append(k.float()); self.hi.append(k.float())
        else:
            self.lo[-1]=torch.minimum(self.lo[-1],k.float())
            self.hi[-1]=torch.maximum(self.hi[-1],k.float())
        self.n+=1

    def output(self,q:torch.Tensor,source:SourceAccess,scaling:float,fraction:float) -> torch.Tensor:
        g,h,d=source.g,q.shape[0],q.shape[1]
        reps=h//g
        lo,hi=torch.stack(self.lo,1),torch.stack(self.hi,1)  # G, pages, d
        count=len(self.lo)
        retain=max(1,min(count,math.ceil(fraction*self.n/self.page)))
        outputs=[]
        for physical in range(g):
            qq=q[physical*reps:(physical+1)*reps].float()
            upper=qq.clamp_min(0)@hi[physical].T + qq.clamp_max(0)@lo[physical].T
            order=torch.argsort(upper,dim=-1,descending=True,stable=True)
            chosen=order[:,:retain]
            # The current page is always visible, in addition to the ranked budget.
            keep=torch.zeros(reps,count,device=q.device,dtype=torch.bool)
            keep.scatter_(1,chosen,True)
            keep[:,-1]=True
            page_ids=torch.nonzero(keep.any(0),as_tuple=False).flatten()
            ids=(page_ids[:,None]*self.page+torch.arange(self.page,device=q.device)).flatten()
            ids=ids[ids<self.n]
            k,v=source.get(physical,ids)
            allowed=keep[:,ids//self.page]
            scores=(qq.double()@k.double().T)*scaling
            scores.masked_fill_(~allowed,-math.inf)
            outputs.append((torch.softmax(scores,-1)@v.double()).to(q.dtype))
        return torch.cat(outputs,0)

    def storage_bytes(self) -> int:
        return sum(nbytes(t) for t in self.lo+self.hi)


class LSHIndex:
    """MagicPIG inclusion estimator with a GPU-resident inverted hash index.

    SimHash collisions must occur in at least two independent tables. The
    inverse inclusion probability appears in the attention weight. Each query
    head uses its own selected set; only raw physical reads are unioned for GQA.
    """
    def __init__(self,keys:torch.Tensor,bits:int,tables:int,seed:int,append_limit:int=64):
        if not 1<=bits<=16 or tables<2:
            raise ValueError("LSH requires 1 <= K <= 16 and L >= 2")
        self.bits,self.tables,self.n=bits,tables,keys.shape[1]
        self.g,self.d=keys.shape[0],keys.shape[-1]
        self.buckets=2**bits
        self.append_limit=append_limit
        generator=torch.Generator(device="cpu").manual_seed(seed)
        # The same projector is allowed across physical heads. Within a head,
        # the K*L hyperplanes are independently sampled.
        self.projector=torch.randn(self.d,bits*tables,generator=generator,dtype=torch.float32).to(keys.device)
        self.bitweights=(2**torch.arange(bits,device=keys.device,dtype=torch.int64))
        eligible=keys[:,4:]
        self.center=(eligible.double().mean(1) if eligible.shape[1] else keys.double().mean(1)).float()
        code_chunks=[]
        for start in range(4,self.n,256):
            code_chunks.append(self.encode(keys[:,start:start+256].float()-self.center[:,None]))
        self.codes=torch.cat(code_chunks,1) if code_chunks else torch.empty(
            self.g,0,tables,device=keys.device,dtype=torch.int64)
        self.base_ids=torch.arange(4,self.n,device=keys.device,dtype=torch.long)
        self.pending:tuple[tuple[int,torch.Tensor],...]=()
        self._rebuild()

    def encode(self,x:torch.Tensor) -> torch.Tensor:
        projected=x@self.projector
        signs=(projected.reshape(*x.shape[:-1],self.tables,self.bits)>0).long()
        return (signs*self.bitweights).sum(-1)

    def _rebuild(self) -> None:
        # Inverted posting lists: one sorted record array per table, with offsets.
        self.postings=[];self.offsets=[]
        for g in range(self.g):
            codes=self.codes[g].T.contiguous()  # L, N
            orders=torch.argsort(codes,dim=-1,stable=True)
            self.postings.append(self.base_ids[orders])
            offset=torch.arange(self.tables,device=codes.device)[:,None]*self.buckets
            counts=torch.bincount((codes+offset).flatten(),minlength=self.tables*self.buckets)
            counts=counts.reshape(self.tables,self.buckets)
            starts=torch.cat([counts.new_zeros(self.tables,1),counts.cumsum(1)],1)
            self.offsets.append(starts)

    def snapshot(self) -> tuple:
        return self.n,self.codes,self.base_ids,self.pending,self.postings,self.offsets

    def restore(self,state:tuple) -> None:
        self.n,self.codes,self.base_ids,self.pending,self.postings,self.offsets=state

    def append(self,k:torch.Tensor,v:torch.Tensor) -> None:
        code=self.encode(k.float()-self.center)
        self.pending=self.pending+((self.n,code),)
        self.n+=1
        if len(self.pending)>=self.append_limit:
            additions=torch.stack([x[1] for x in self.pending],1)
            ids=torch.tensor([x[0] for x in self.pending],device=k.device,dtype=torch.long)
            self.codes=torch.cat([self.codes,additions],1)
            self.base_ids=torch.cat([self.base_ids,ids])
            self.pending=()
            self._rebuild()

    def lookup(self,g:int,query:torch.Tensor) -> torch.Tensor:
        code=self.encode(query.float())  # L
        ts=torch.arange(self.tables,device=query.device)
        starts=self.offsets[g][ts,code]
        ends=self.offsets[g][ts,code+1]
        lengths=ends-starts
        table_ids=torch.repeat_interleave(ts,lengths)
        if table_ids.numel():
            prefix=lengths.cumsum(0)-lengths
            positions=torch.arange(table_ids.numel(),device=query.device)-torch.repeat_interleave(prefix,lengths)
            positions=positions+torch.repeat_interleave(starts,lengths)
            postings=self.postings[g][table_ids,positions]
            collisions=torch.bincount(postings,minlength=self.n)
        else:
            collisions=torch.zeros(self.n,device=query.device,dtype=torch.long)
        if self.pending:
            pending_codes=torch.stack([x[1][g] for x in self.pending])
            matches=(pending_codes==code[None]).sum(-1)
            pending_ids=torch.tensor([x[0] for x in self.pending],device=query.device)
            collisions[pending_ids]=matches
        selected=collisions>=2
        # Fixed deterministic strata: first four and most recent 64 records.
        selected[:min(4,self.n)]=True
        selected[max(0,self.n-64):]=True
        return selected

    @staticmethod
    def inclusion(cosine:torch.Tensor,bits:int,tables:int) -> torch.Tensor:
        p=1-torch.acos(cosine.clamp(-1,1))/math.pi
        a=p.pow(bits)
        lq=torch.log1p(-a)
        prob=-torch.expm1(tables*lq)-tables*a*torch.exp((tables-1)*lq)
        # A series avoids cancellation at extremely small collision probability.
        small=a<1e-8/tables
        series=0.5*tables*(tables-1)*a*a*(1-(2/3)*(tables-2)*a)
        return torch.where(small,series,prob).clamp(0,1)

    def output(self,q:torch.Tensor,source:SourceAccess,scaling:float) -> torch.Tensor:
        reps=q.shape[0]//self.g
        out=[]
        for g in range(self.g):
            qq=q[g*reps:(g+1)*reps]
            masks=torch.stack([self.lookup(g,x) for x in qq])
            ids=torch.nonzero(masks.any(0),as_tuple=False).flatten()
            k,v=source.get(g,ids)
            kd=k.double()-self.center[g].double()
            qd=qq.double()
            qnorm=qd.norm(dim=-1)[:,None]
            knorm=kd.norm(dim=-1)[None]
            norms=qnorm*knorm
            cos=(qd@kd.T)/norms.clamp_min(1e-300)
            cos=torch.where(norms==0,torch.zeros_like(cos),cos)
            # Our sign(0) convention hashes two zero vectors identically.
            cos=torch.where((qnorm==0)&(knorm==0),torch.ones_like(cos),cos)
            prob=self.inclusion(cos,self.bits,self.tables)
            fixed=(ids<4)|(ids>=self.n-64)
            prob[:,fixed]=1
            allowed=masks[:,ids]
            if bool(torch.any((prob<=0)&allowed)):
                raise ArithmeticError("A selected LSH record has nonpositive inclusion probability")
            # Key centering changes each head's logits by one common shift.
            scores=qd@k.double().T*scaling-torch.log(prob.clamp_min(1e-300))
            scores.masked_fill_(~allowed,-math.inf)
            out.append((torch.softmax(scores,-1)@v.double()).to(q.dtype))
        return torch.cat(out,0)

    def storage_bytes(self) -> int:
        tensors=[self.projector,self.bitweights,self.center,self.codes,self.base_ids]
        tensors+=self.postings+self.offsets+[x[1] for x in self.pending]
        return sum(nbytes(t) for t in tensors)


class TensorOperandCounter:
    """Optional logical operand accounting, outside all latency measurements."""
    def __init__(self,source_tensors:Sequence[torch.Tensor]=()):
        from torch.utils._python_dispatch import TorchDispatchMode
        parent=self
        self.source_ptrs={t.untyped_storage().data_ptr() for t in source_tensors}
        self.source_operands=0
        self.auxiliary_operands=0
        self.write_operands=0
        views={"alias","detach","view","_unsafe_view","reshape","transpose","t",
               "permute","slice","select","as_strided","expand","squeeze","unsqueeze"}
        def tensors(x:Any) -> Iterator[torch.Tensor]:
            if torch.is_tensor(x):yield x
            elif isinstance(x,dict):
                for v in x.values():yield from tensors(v)
            elif isinstance(x,(list,tuple)):
                for v in x:yield from tensors(v)
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs={} if kwargs is None else kwargs
                name=func._schema.name.split("::")[-1]
                out=func(*args,**kwargs)
                if name in views:return out
                for t in tensors((args,kwargs)):
                    if t.layout!=torch.strided:continue
                    if t.untyped_storage().data_ptr() in parent.source_ptrs:
                        parent.source_operands+=nbytes(t)
                    else:parent.auxiliary_operands+=nbytes(t)
                for t in tensors(out):parent.write_operands+=nbytes(t)
                return out
        self.mode=Mode()

    def __enter__(self):self.mode.__enter__();return self
    def __exit__(self,*args):return self.mode.__exit__(*args)


@dataclasses.dataclass
class Emission:
    token:int
    source:SourceAccess
    early:bool=False
    pre_support:bool=False
    stages:int=0
    allocated:float=0.0
    probabilities:Optional[torch.Tensor]=None
    logits:Optional[torch.Tensor]=None
    trace:list[dict]=dataclasses.field(default_factory=list)
    auxiliary_operand_bytes:Optional[int]=None


class TokenReady(Exception):
    def __init__(self,result:Emission):
        super().__init__("last-layer token selected")
        self.result=result


class LastLayerRunner:
    def __init__(self,model:torch.nn.Module,tokenizer:Any,cfg:Config):
        self.model,self.tokenizer,self.cfg=model,tokenizer,cfg
        self.device=next(model.parameters()).device
        self.dtype=next(model.parameters()).dtype
        self.block=model.model.layers[-1]
        self.attention=self.block.self_attn
        self.last_layer=int(self.attention.layer_idx)
        self.mode="prefill"
        self.method=Method("dense")
        self.policy=Policy("topk",cfg.top_k,cfg.temperature)
        self.rng=np.random.default_rng(17)
        self.aux:Any=None
        self.last_state:Optional[dict]=None
        self.h:Optional[torch.Tensor]=None
        self.state:Any=None
        self.instrument=False
        self.counter:Optional[TensorOperandCounter]=None
        self._hooks=[]
        self._install()

    def _install(self) -> None:
        from transformers import AttentionInterface,AttentionMaskInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.masking_utils import sdpa_mask
        self.native_attention=sdpa_attention_forward
        self.backend="rea_benchmark_"+str(id(self))
        AttentionInterface.register(self.backend,self.attention_entry)
        # Essential: omitting the mask registration silently removes causal masks
        # from chunked/padded prefill in recent Transformers versions.
        AttentionMaskInterface.register(self.backend,sdpa_mask)
        self.model.set_attn_implementation(self.backend)
        def before_block(module,args,kwargs):
            hidden=kwargs.get("hidden_states",args[0] if args else None)
            if hidden is None:raise RuntimeError("Missing final-block residual input")
            self.h=hidden[:,-1:,:]
        self._hooks.append(self.block.register_forward_pre_hook(before_block,with_kwargs=True))

    def close(self) -> None:
        for h in self._hooks:h.remove()
        self._hooks=[]
        self.model.set_attn_implementation("sdpa")
        # Registry cleanup prevents successive checkpoint runs retaining models.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        with contextlib.suppress(Exception):
            del ALL_ATTENTION_FUNCTIONS._global_mapping[self.backend]
        with contextlib.suppress(Exception):
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            del ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[self.backend]

    def attention_entry(self,module,query,key,value,attention_mask,**kwargs):
        if module.layer_idx!=self.last_layer:
            return self.native_attention(module,query,key,value,attention_mask,**kwargs)
        self.last_state={"module":module,"q":query,"k":key,"v":value,
                         "mask":attention_mask,"kwargs":dict(kwargs),"h":self.h}
        if self.mode=="prefill":
            return self.native_attention(module,query,key,value,attention_mask,**kwargs)
        if query.shape[0]!=1 or query.shape[-2]!=1:
            raise ValueError("The measured sampler supports batch one, one-token decoding.")
        if kwargs.get("sliding_window") not in (None,0):
            raise ValueError("Sliding-window final layers require a separate source interface")
        ctx=self.last_state
        source=SourceAccess(key[0],value[0])
        def computation() -> Emission:
            self.update_aux(ctx,source)
            return self.decode(ctx,source,self.method,self.policy,self.rng)
        if self.instrument:
            with TensorOperandCounter((key,value)) as counter:
                result=computation()
            result.auxiliary_operand_bytes=counter.auxiliary_operands+counter.write_operands
        else:result=computation()
        raise TokenReady(result)

    def suffix(self,attention_output:torch.Tensor,h:torch.Tensor) -> torch.Tensor:
        o=attention_output.reshape(1,1,-1).to(self.dtype)
        u=h+self.attention.o_proj(o)
        z=u+self.block.mlp(self.block.post_attention_layernorm(u))
        return self.model.lm_head(self.model.model.norm(z)).reshape(-1)

    def suffix_box(self,attention_box:Box,h:torch.Tensor) -> Box:
        x=Box(attention_box.lo.reshape(1,-1),attention_box.hi.reshape(1,-1))
        projected=linear_box(x,self.attention.o_proj,self.dtype,self.cfg.interval_chunk)
        u=model_round(projected+Box.point(h.reshape(1,-1)),self.dtype)
        norm=self.block.post_attention_layernorm
        eps=float(getattr(norm,"variance_epsilon",self.model.config.rms_norm_eps))
        n=rms_box(u,norm.weight,eps,self.dtype)
        gate=linear_box(n,self.block.mlp.gate_proj,self.dtype,self.cfg.interval_chunk)
        value=linear_box(n,self.block.mlp.up_proj,self.dtype,self.cfg.interval_chunk)
        gated=model_round(silu_box(gate,self.dtype)*value,self.dtype)
        ff=linear_box(gated,self.block.mlp.down_proj,self.dtype,self.cfg.interval_chunk)
        z=model_round(u+ff,self.dtype)
        norm2=self.model.model.norm
        eps2=float(getattr(norm2,"variance_epsilon",self.model.config.rms_norm_eps))
        final=rms_box(z,norm2.weight,eps2,self.dtype)
        return linear_box(final,self.model.lm_head,self.dtype,self.cfg.interval_chunk)

    def dense_logits(self,ctx:dict,source:Optional[SourceAccess]=None) -> torch.Tensor:
        if source is not None:source.dense()
        o,_=self.native_attention(ctx["module"],ctx["q"],ctx["k"],ctx["v"],ctx["mask"],**ctx["kwargs"])
        return self.suffix(o,ctx["h"])

    def prepare(self,input_prefix:Sequence[int],method:Method,hash_seed:int,max_new:int) -> dict:
        from transformers import DynamicCache
        if len(input_prefix)<2:raise ValueError("A prefix needs at least two tokens.")
        self.mode="prefill";self.aux=None;self.method=method
        self.state=DynamicCache(config=self.model.config)
        ids=torch.tensor([list(input_prefix[:-1])],device=self.device,dtype=torch.long)
        sync(self.device);t0=time.perf_counter()
        # One prefill keeps its native causal attention, including RoPE scaling.
        with torch.inference_mode():
            self.model.model(input_ids=ids,past_key_values=self.state,use_cache=True,return_dict=True)
        sync(self.device);prefill_ms=(time.perf_counter()-t0)*1000
        if self.last_state is None:raise RuntimeError("The custom attention interface was not reached")
        k,v=self.last_state["k"][0],self.last_state["v"][0]
        sync(self.device);t0=time.perf_counter()
        with torch.inference_mode():
            if not method.is_dense:
                if method.kind=="progressive":
                    self.aux=SummaryTree(k,v,self.cfg.leaf,self.cfg.initial,method.degree,len(input_prefix)+max_new+2)
                elif method.kind=="quest":self.aux=PageIndex(k,self.cfg.page)
                elif method.kind=="magicpig":self.aux=LSHIndex(k,method.bits,method.tables,hash_seed)
                else:raise ValueError(method.kind)
        sync(self.device);construction_ms=(time.perf_counter()-t0)*1000
        storage=self.aux.storage_bytes() if self.aux is not None else 0
        return {"prefill_ms":prefill_ms,"construction_ms":construction_ms,
                "auxiliary_retained_bytes":storage,"source_retained_bytes":nbytes(k)+nbytes(v)}

    def update_aux(self,ctx:dict,source:Optional[SourceAccess]=None) -> None:
        if self.aux is None:return
        n=ctx["k"].shape[-2]
        if self.aux.n==n-1:
            if source is not None:source.append_access()
            self.aux.append(ctx["k"][0,:,-1],ctx["v"][0,:,-1])
        elif self.aux.n!=n:raise RuntimeError(f"Auxiliary length {self.aux.n} differs from source {n}")

    def crop(self,n:int) -> None:
        self.state.crop(n)

    def step(self,token:int,method:Method,policy:Policy,rng:np.random.Generator,
             instrument:bool=False) -> Emission:
        self.mode="sample";self.method=method;self.policy=policy;self.rng=rng;self.instrument=instrument
        ids=torch.tensor([[token]],device=self.device,dtype=torch.long)
        try:
            with torch.inference_mode():
                self.model.model(input_ids=ids,past_key_values=self.state,use_cache=True,return_dict=True)
        except TokenReady as done:
            return done.result
        finally:self.instrument=False
        raise RuntimeError("The model bypassed the last-layer attention interception")

    def decode(self,ctx:dict,source:SourceAccess,method:Method,policy:Policy,
               rng:np.random.Generator,force_defer:bool=False) -> Emission:
        if method.is_dense:
            logits=self.dense_logits(ctx,source);p=policy_prob(logits,policy)
            return Emission(draw_categorical(p,rng),source,probabilities=p,logits=logits)
        q=ctx["q"][0,:,0]
        scale=float(ctx["kwargs"].get("scaling",self.attention.scaling))
        if method.kind in {"quest","magicpig"}:
            o=(self.aux.output(q,source,scale,method.fraction) if method.kind=="quest"
               else self.aux.output(q,source,scale))
            logits=self.suffix(o,ctx["h"]);p=policy_prob(logits,policy)
            return Emission(draw_categorical(p,rng),source,probabilities=p,logits=logits)
        return self.progressive(ctx,source,method,policy,rng,force_defer)

    def progressive(self,ctx:dict,source:SourceAccess,method:Method,policy:Policy,
                    rng:np.random.Generator,force_defer:bool) -> Emission:
        q=ctx["q"][0,:,0]
        scale=float(ctx["kwargs"].get("scaling",self.attention.scaling))
        frontier=Frontier(self.aux,q,source,scale)
        alpha=torch.zeros(self.model.config.vocab_size,device=q.device,dtype=torch.float64)
        trace=[];support_seen=False
        for stage in range(method.stages):
            attention_box=frontier.enclosure(self.dtype)
            logits_box=self.suffix_box(attention_box,ctx["h"])
            cert=probability_certificate(logits_box,policy)
            support_seen=support_seen or cert.support
            offered=cert.lower if (not method.support_first or support_seen) else torch.zeros_like(alpha)
            new_alpha=torch.maximum(alpha,offered)
            previous=float(alpha.sum());current=float(new_alpha.sum())
            if current>1+self.cfg.law_tol:raise ArithmeticError("Accumulated probability exceeds one")
            delta=(new_alpha-alpha).clamp_min(0)
            # Keep full endpoints only in untimed diagnostics.
            record={"stage":stage,"C":current,"support":support_seen,
                    "source_bytes":source.bytes,"reads":source.distinct if force_defer else -1,
                    "attention_radius":frontier.last_radius,
                    "logit_width":float((logits_box.hi-logits_box.lo).max())/policy.temperature}
            if force_defer:
                record.update({"lower_logits":logits_box.lo.flatten(),
                               "upper_logits":logits_box.hi.flatten(),
                               "alpha":new_alpha.clone(),"core":cert.core})
            trace.append(record)
            emit_prob=(current-previous)/max(1-previous,1e-300)
            if not force_defer and rng.random()<min(1.0,max(0.0,emit_prob)):
                token=draw_categorical(delta,rng)
                return Emission(token,source,True,not support_seen,stage+1,current,trace=trace)
            alpha=new_alpha
            if stage+1<method.stages and not frontier.refine():break
        logits=self.dense_logits(ctx,source)
        p=policy_prob(logits,policy)
        residual=p-alpha
        excess=float((-residual).clamp_min(0).max())
        if excess>self.cfg.law_tol:
            raise ArithmeticError(f"Certificate exceeds dense target by {excess:.3e}; run invalid")
        residual=residual.clamp_min(0)
        current=float(alpha.sum())
        if float(residual.sum())<=0:
            # This branch has zero mathematical mass. It occurs only in a fully
            # deferred diagnostic, so the target supplies its diagnostic token.
            if not force_defer:raise ArithmeticError("A zero-mass completion branch was reached")
            token=draw_categorical(p,rng)
        else:token=draw_categorical(residual,rng)
        return Emission(token,source,False,False,len(trace),current,p,logits,trace)

    def audit(self,ctx:dict,method:Method,policy:Policy,dense_logits:torch.Tensor) -> dict:
        source=SourceAccess(ctx["k"][0],ctx["v"][0])
        if not method.is_dense:source.append_access()
        ref=policy_prob(dense_logits,policy)
        result=self.decode(ctx,source,method,policy,np.random.default_rng(999),force_defer=True)
        if method.kind!="progressive" or method.is_dense:
            discrepancy=tv(result.probabilities,ref)
            return {"conditional_tv":discrepancy,"certificate_excess":0.0,"logit_violations":0,
                    "support_violations":0,"normalization_error":abs(float(result.probabilities.sum())-1),
                    "expected_early":None,"expected_pre_support":None,
                    "expected_reads":source.distinct,"expected_source_bytes":source.bytes,
                    "expected_stages":0.0,"audit_stages":[]}
        allocated=torch.zeros_like(ref)
        reconstructed=torch.zeros_like(ref)
        previous_C=0.0;previous_reads=0;previous_bytes=0
        expected_reads=0.0;expected_bytes=0.0;expected_stages=0.0;pre_support=0.0
        logit_bad=0;support_bad=0;excess=0.0;stages=[]
        z=dense_logits.double().flatten()
        for stage in result.trace:
            a=stage["alpha"]
            survival=1-previous_C
            expected_reads+=survival*(stage["reads"]-previous_reads)
            expected_bytes+=survival*(stage["source_bytes"]-previous_bytes)
            expected_stages+=survival
            reconstructed+=a-allocated
            logit_bad+=int(((z<stage["lower_logits"]-self.cfg.logit_tol)|
                            (z>stage["upper_logits"]+self.cfg.logit_tol)).sum())
            excess=max(excess,float((a-ref).clamp_min(0).max()))
            support_bad+=int((stage["core"]&(ref==0)).sum())
            if not stage["support"]:pre_support=stage["C"]
            stages.append({k:v for k,v in stage.items() if k not in {"alpha","core","lower_logits","upper_logits"}})
            allocated=a;previous_C=stage["C"]
            previous_reads=stage["reads"];previous_bytes=stage["source_bytes"]
        expected_reads+=(1-previous_C)*(source.distinct-previous_reads)
        expected_bytes+=(1-previous_C)*(source.bytes-previous_bytes)
        remainder=(ref-allocated).clamp_min(0)
        if float(remainder.sum())>0:
            reconstructed+=(1-previous_C)*remainder/remainder.sum()
        law_tv=tv(reconstructed,ref)
        norm_error=abs(float(reconstructed.sum())-1)
        if logit_bad or support_bad or excess>self.cfg.law_tol or law_tv>self.cfg.law_tol:
            raise ArithmeticError(f"Audit failed: logit={logit_bad}, support={support_bad}, "
                                  f"excess={excess:.3e}, law TV={law_tv:.3e}")
        return {"conditional_tv":law_tv,"certificate_excess":excess,"logit_violations":logit_bad,
                "support_violations":support_bad,"normalization_error":norm_error,
                "expected_early":previous_C,"expected_pre_support":pre_support,
                "expected_reads":expected_reads,"expected_source_bytes":expected_bytes,
                "expected_stages":expected_stages,"audit_stages":stages}


def _numeric_stack_preflight() -> None:
    """Fail early on a NumPy/SciPy ABI/version mismatch before Transformers imports sklearn."""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        return
    try:
        scipy_version=importlib.metadata.version("scipy")
        requirements=importlib.metadata.requires("scipy") or []
    except importlib.metadata.PackageNotFoundError:
        return
    numpy_version=Version(np.__version__)
    bad=[]
    for raw in requirements:
        try:
            req=Requirement(raw)
        except Exception:
            continue
        if req.name.lower()!="numpy":
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        if req.specifier and numpy_version not in req.specifier:
            bad.append(str(req.specifier))
    if bad:
        raise RuntimeError(
            f"Incompatible numerical stack: numpy=={np.__version__}, scipy=={scipy_version}, "
            f"while this SciPy build requires NumPy {bad[0]}. For this benchmark use a "
            "consistent environment, e.g. `python -m pip install -U 'numpy>=1.26,<2.3' "
            "'scipy>=1.11' 'scikit-learn>=1.4'`, then restart Python."
        )


def _transformers_api_preflight(transformers:Any) -> None:
    """Check the public APIs used by this benchmark instead of pinning one release string."""
    try:
        from packaging.version import Version
        version=Version(transformers.__version__)
        if version < Version("4.57.1") or version >= Version("6"):
            raise RuntimeError(
                f"Unsupported transformers=={transformers.__version__}. "
                "Use transformers>=4.57.1,<6 for this benchmark."
            )
    except ImportError:
        pass
    missing=[]
    for name in ("AttentionInterface","AttentionMaskInterface"):
        if not hasattr(transformers,name):
            missing.append(name)
    try:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: F401
        from transformers.masking_utils import sdpa_mask  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"transformers=={transformers.__version__} lacks the SDPA attention/mask API required "
            "by this benchmark."
        ) from exc
    if missing:
        raise RuntimeError(
            f"transformers=={transformers.__version__} is missing required public APIs: "
            +", ".join(missing)
        )


def _require_safe_pyarrow() -> str:
    """Require Arrow's interpreter-finalization protection for streaming Parquet."""
    try:
        from packaging.version import Version
        version=importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "FineWeb streaming requires pyarrow>=25.0.0. Install it with "
            "`python -m pip install -U 'pyarrow>=25'`."
        ) from exc
    except ImportError:
        version=importlib.metadata.version("pyarrow")
        major=int(version.split('.',1)[0])
        if major < 25:
            raise RuntimeError(
                f"Detected pyarrow=={version}; FineWeb streaming requires pyarrow>=25.0.0."
            )
        return version
    if Version(version)<Version("25.0.0"):
        raise RuntimeError(
            f"Detected pyarrow=={version}. FineWeb streaming requires pyarrow>=25.0.0 to avoid "
            "the interpreter-finalization crash seen with partially consumed Parquet streams. "
            "Upgrade with `python -m pip install -U 'pyarrow>=25'`, then restart Python."
        )
    return version


def _close_stream_while_runtime_alive(*objects:Any) -> None:
    """Drop generator-backed dataset objects before CPython starts finalizing."""
    for obj in objects:
        if obj is None:
            continue
        close=getattr(obj,"close",None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
    gc.collect()


def load_checkpoint(model_id:str,cfg:Config) -> tuple[Any,Any,dict]:
    _numeric_stack_preflight()
    try:
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers>=4.57.1,<6 and the dependencies in the file header") from exc
    if cfg.allow_other_transformers:
        LOG.warning("--allow-other-transformers bypasses the tested release range; public API compatibility is still checked")
        try:
            from transformers import AttentionInterface,AttentionMaskInterface  # noqa: F401
            from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: F401
            from transformers.masking_utils import sdpa_mask  # noqa: F401
        except Exception as exc:
            raise RuntimeError("The installed Transformers build lacks the attention APIs required by this benchmark") from exc
    else:
        _transformers_api_preflight(transformers)
    device=torch.device(cfg.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for functional checks, not GPU tables.")
    if cfg.dtype not in {"bfloat16","float32"}:
        raise ValueError("Use bfloat16 for the paper or float32 for CPU integration checks")
    dtype=getattr(torch,cfg.dtype)
    if device.type=="cuda":
        torch.cuda.set_device(device)
        if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16. Use --dtype float32 and record that configuration.")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if hasattr(torch.backends.cuda.matmul,"allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.set_grad_enabled(False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.manual_seed(17)
    tok=AutoTokenizer.from_pretrained(model_id,revision=cfg.revision,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=cfg.revision,
        torch_dtype=dtype,attn_implementation="sdpa",trust_remote_code=False,
        low_cpu_mem_usage=True).to(device).eval()
    if model.config.model_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported model type {model.config.model_type}; Qwen3 and Llama are implemented")
    for required in ("gate_proj","up_proj","down_proj"):
        if not hasattr(model.model.layers[-1].mlp,required):raise ValueError("Unsupported feed-forward block")
    if getattr(model.config,"hidden_act","silu")!="silu":raise ValueError("The certificate requires SiLU")
    meta={"model":model_id,"resolved_revision":getattr(model.config,"_commit_hash",None),
          "model_type":model.config.model_type,"parameter_count":sum(p.numel() for p in model.parameters()),
          "layers":model.config.num_hidden_layers,"query_heads":model.config.num_attention_heads,
          "kv_heads":model.config.num_key_value_heads,"head_dim":model.model.layers[-1].self_attn.head_dim,
          "dtype":str(dtype),"vocabulary":model.config.vocab_size,
          "transformers":transformers.__version__,"torch":torch.__version__,
          "python":platform.python_version(),"device":str(device),
          "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
          "certificate_status":CERTIFICATE_STATUS}
    return model,tok,meta


def validate_native_adapter(model:Any,tok:Any,cfg:Config) -> LastLayerRunner:
    """Abort before datasets if the intercepted dense suffix changes native logits."""
    ids=tok("A careful experiment compares the same conditional distribution.",return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
    model.set_attn_implementation("sdpa")
    with torch.inference_mode():
        native=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].detach()
    runner=LastLayerRunner(model,tok,cfg)
    runner.prepare(ids[0].tolist(),Method("dense"),17,2)
    out=runner.step(int(ids[0,-1]),Method("dense"),Policy("topk",min(cfg.top_k,native.numel()),cfg.temperature),np.random.default_rng(17))
    # Full prefill and one-token incremental kernels can differ numerically.
    # Check an exact same-shape native incremental execution as the primary check.
    from transformers import DynamicCache
    model.set_attn_implementation("sdpa")
    state=DynamicCache(config=model.config)
    with torch.inference_mode():
        model.model(input_ids=ids[:,:-1],past_key_values=state,use_cache=True)
        native_incremental=model(input_ids=ids[:,-1:],past_key_values=state,use_cache=True,logits_to_keep=1).logits[0,-1].detach()
    model.set_attn_implementation(runner.backend)
    maxerr=float((out.logits.float()-native_incremental.float()).abs().max())
    full_tv=tv(torch.softmax(out.logits.double(),0),torch.softmax(native_incremental.double(),0))
    if maxerr>1e-6 or full_tv>1e-10:
        runner.close()
        raise RuntimeError(f"Native incremental adapter mismatch: max logit error={maxerr}, TV={full_tv}")
    LOG.info("Native same-shape dense adapter check passed. Prefill/incremental max difference: %.6g",
             float((native.float()-native_incremental.float()).abs().max()))
    return runner


# --------------------------- DATA PREPARATION --------------------------

def fetch_url(url:str,destination:Path) -> Path:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():return destination
    request=urllib.request.Request(url,headers={"User-Agent":"rea-research-benchmark/1.0"})
    temp=destination.with_suffix(destination.suffix+".partial")
    with urllib.request.urlopen(request,timeout=120) as response,temp.open("wb") as out:
        shutil.copyfileobj(response,out)
    temp.replace(destination)
    return destination


def github_commit(repo:str,revision:str,folder:Path) -> str:
    info=folder/(slug(repo)+"_"+slug(revision)+".json")
    if not info.exists():fetch_url(f"https://api.github.com/repos/{repo}/commits/{revision}",info)
    content=json.loads(info.read_text())
    commit=content.get("sha")
    if not commit or not re.fullmatch("[0-9a-f]{40}",commit):raise RuntimeError("Unable to pin official source revision")
    return commit


def hub_revision(repo:str,revision:str,kind:str) -> str:
    from huggingface_hub import HfApi
    return HfApi().repo_info(repo_id=repo,repo_type=kind,revision=revision).sha


def encode_chat(tok:Any,text:str,assistant_prefix:str="") -> list[int]:
    if not tok.chat_template:raise ValueError("The checkpoint does not provide a chat template")
    ids=tok.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True)
    if assistant_prefix:ids+=tok.encode(assistant_prefix,add_special_tokens=False)
    return list(ids)


def middle_truncated_prompt(tok:Any,template:str,record:dict,cap:int) -> list[int]:
    def encoded(context:str) -> list[int]:
        fields=dict(record);fields["context"]=context
        return encode_chat(tok,template.format(**fields))
    context=record.get("context","")
    ids=encoded(context)
    if len(ids)<=cap:return ids
    body=tok.encode(context,add_special_tokens=False)
    if len(encoded(""))>cap:
        raise ValueError("Task instruction and question exceed the context cap")
    lo,hi=0,len(body)
    best=encoded("")
    while lo<=hi:
        retained=(lo+hi)//2
        front=(retained+1)//2;back=retained//2
        kept=body[:front]+(body[-back:] if back else [])
        candidate=encoded(tok.decode(kept,skip_special_tokens=False))
        if len(candidate)<=cap:best=candidate;lo=retained+1
        else:hi=retained-1
    return best


def safe_extract(zip_path:Path,dest:Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        root=dest.resolve()
        for member in archive.infolist():
            target=(dest/member.filename).resolve()
            if target!=root and root not in target.parents:
                raise RuntimeError("Unsafe archive path")
            # Reject symlink members.
            if ((member.external_attr>>16)&0o170000)==0o120000:
                raise RuntimeError("Symbolic links are not accepted in benchmark archives")
        archive.extractall(dest)


def official_ruler_repo(cfg:Config) -> tuple[Path,str]:
    root=Path(cfg.out)/"data"/"official_sources"
    commit=github_commit("NVIDIA/RULER",cfg.ruler_revision,root)
    folder=root/f"RULER-{commit}"
    if not folder.exists():
        archive=fetch_url(f"https://codeload.github.com/NVIDIA/RULER/zip/{commit}",root/f"ruler-{commit}.zip")
        safe_extract(archive,root)
    data=folder/"scripts"/"data"
    essay=data/"synthetic"/"json"/"PaulGrahamEssays.json"
    if not essay.exists():
        download=data/"synthetic"/"json"/"download_paulgraham_essay.py"
        subprocess.run([sys.executable,str(download)],cwd=download.parent,check=True)
    return folder,commit


def prepare_ruler_records(cfg:Config,tokenizers:dict[str,Any]) -> dict[tuple[int,str,str],list[dict]]:
    """Use one official raw task instance across checkpoints, preserving answers.

    Context labels denote upper limits. Generator input lengths are reduced, when
    necessary, until ALL model-specific chat encodings fit. The actual encoded
    lengths are recorded, so model-dependent tokenization is visible.
    """
    result={}
    source_root=Path(cfg.ruler_dir) if cfg.ruler_dir else Path(cfg.out)/"data"/"ruler_raw"
    repo=commit=None
    if cfg.prepare_ruler:repo,commit=official_ruler_repo(cfg)
    for context in cfg.contexts:
        for split,n in (("calibration",cfg.n_cal),("evaluation",cfg.n_eval)):
            for task in RULER_TASKS:
                raw_path=source_root/str(context)/split/f"{task}.jsonl"
                records=jsonl_read(raw_path)
                if records:
                    if len(records)!=n:raise RuntimeError(f"Expected {n} official rows at {raw_path}")
                elif not cfg.prepare_ruler:
                    raise FileNotFoundError(f"Missing {raw_path}. Supply --ruler-dir or --prepare-ruler.")
                else:
                    assert repo is not None
                    generation_cap=context+128-64
                    seed=int(stable_seed(cfg.data_seed,context,split,task)%(2**31-1))
                    for attempt in range(12):
                        work=Path(cfg.out)/"data"/"ruler_generation"/str(context)/split/task/f"try{attempt}"
                        command=[sys.executable,str(repo/"scripts/data/prepare.py"),
                            "--save_dir",str(work.resolve()),"--benchmark","synthetic","--task",task,
                            "--tokenizer_path",cfg.models[0],"--tokenizer_type","hf",
                            "--max_seq_length",str(generation_cap),"--model_template_type","base",
                            "--num_samples",str(n),"--random_seed",str(seed),"--subset","validation"]
                        work.mkdir(parents=True,exist_ok=True)
                        with (work/"generator.log").open("w") as log:
                            subprocess.run(command,cwd=repo/"scripts",stdout=log,stderr=subprocess.STDOUT,check=True)
                        generated=work/task/"validation.jsonl"
                        records=jsonl_read(generated)
                        # The official wrapper may report a child failure without
                        # returning a nonzero exit status; validate its actual file.
                        if len(records)!=n:
                            raise RuntimeError(f"Official RULER generation failed; inspect {work/'generator.log'}")
                        lengths=[len(encode_chat(tok,row["input"],row.get("answer_prefix","")))
                                 for row in records for tok in tokenizers.values()]
                        longest=max(lengths)
                        if longest<=context:break
                        generation_cap=max(512,int((generation_cap-128)*context/longest)-64+128)
                    else:raise RuntimeError("Official RULER prompts did not fit all tokenizer-specific caps")
                    raw_path.parent.mkdir(parents=True,exist_ok=True)
                    with raw_path.open("w",encoding="utf-8") as f:
                        for row in records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
                    atomic_json(raw_path.with_suffix(".source.json"),{"repository":"NVIDIA/RULER",
                        "revision":commit,"task":task,"seed":seed,"generator_cap":generation_cap,
                        "input_cap":context,"same_raw_instances_across_models":True})
                for row in records:
                    if not isinstance(row.get("input"),str) or not isinstance(row.get("outputs"),list):
                        raise ValueError(f"Invalid official RULER row in {raw_path}")
                result[context,split,task]=records
    return result


def data_signature(cfg:Config) -> str:
    keys=("models","revision","datasets","contexts","n_eval","n_cal","max_long_examples",
          "teacher_tokens","data_seed","ruler_revision","fineweb_revision","longbench_revision")
    return hashlib.sha256(json.dumps({k:getattr(cfg,k) for k in keys},sort_keys=True).encode()).hexdigest()


def prepare_data(cfg:Config) -> dict[str,list[dict]]:
    root=Path(cfg.out)/"data"
    done=root/"data_index.json"
    signature=data_signature(cfg)
    if done.exists():
        old=json.loads(done.read_text())
        if old["signature"]!=signature:
            raise RuntimeError("Existing data belong to a different configuration. Choose a new --out directory.")
        return {m:jsonl_read(root/(slug(m)+".jsonl")) for m in cfg.models}
    from transformers import AutoTokenizer
    tokenizers={m:AutoTokenizer.from_pretrained(m,revision=cfg.revision,trust_remote_code=False) for m in cfg.models}
    data={m:[] for m in cfg.models}
    source_meta={}
    def add(model:str,dataset:str,task:str,context:int,split:str,prompt_id:str,
            ids:list[int],answers:list[str],max_new:int,targets:Optional[list[int]]=None,
            **extra:Any) -> None:
        if len(ids)>context:raise ValueError(f"Prompt {prompt_id} exceeds context cap")
        data[model].append({"model":model,"dataset":dataset,"task":task,"context":context,
            "split":split,"prompt_id":prompt_id,"input_ids":ids,"answers":answers,
            "max_new_tokens":max_new,"teacher_targets":targets or [],"actual_context":len(ids),**extra})
    if "fineweb" in cfg.datasets:
        global _USED_HF_STREAMING
        from datasets import load_dataset
        pyarrow_version=_require_safe_pyarrow()
        revision=hub_revision("HuggingFaceFW/fineweb",cfg.fineweb_revision,"dataset")
        source_meta["fineweb"]={"revision":revision,"configuration":"sample-10BT",
                                "pyarrow_version":pyarrow_version}
        fineweb=None
        stream=None
        try:
            fineweb=load_dataset("HuggingFaceFW/fineweb",name="sample-10BT",split="train",
                                 revision=revision,streaming=True)
            fineweb=fineweb.shuffle(seed=cfg.data_seed,buffer_size=10000)
            stream=iter(fineweb)
            _USED_HF_STREAMING=True
            seen=set()
            for context in cfg.contexts:
                for split,n in (("calibration",cfg.n_cal),("evaluation",cfg.n_eval)):
                    LOG.info("Preparing FineWeb %s context=%d examples=%d",split,context,n)
                    for _ in range(n):
                        packed={m:[] for m in cfg.models};docs=[]
                        while min(map(len,packed.values()))<context+cfg.teacher_tokens:
                            row=next(stream)
                            sample_text=row.get("text","")
                            content_hash=hashlib.sha256(sample_text.encode()).hexdigest()
                            doc_id=str(row.get("id",row.get("url",content_hash)))+":"+content_hash
                            if not sample_text.strip() or doc_id in seen:continue
                            seen.add(doc_id);docs.append(doc_id)
                            for model,tok in tokenizers.items():
                                if packed[model] and tok.eos_token_id is not None:
                                    packed[model].append(tok.eos_token_id)
                                packed[model].extend(tok.encode(sample_text,add_special_tokens=False))
                        pid="fineweb_"+hashlib.sha256("\n".join(docs).encode()).hexdigest()[:24]
                        for model,ids in packed.items():
                            add(model,"FineWeb","continuation",context,split,pid,ids[:context],[],128,
                                ids[context:context+cfg.teacher_tokens],document_ids=docs)
        finally:
            _close_stream_while_runtime_alive(stream,fineweb)
            stream=None
            fineweb=None
    if "longbench" in cfg.datasets:
        from huggingface_hub import hf_hub_download
        revision=hub_revision("THUDM/LongBench",cfg.longbench_revision,"dataset")
        archive=hf_hub_download(repo_id="THUDM/LongBench",filename="data.zip",repo_type="dataset",revision=revision)
        commit=github_commit("THUDM/LongBench","main",root/"official_sources")
        prompts_file=fetch_url(f"https://raw.githubusercontent.com/THUDM/LongBench/{commit}/LongBench/config/dataset2prompt.json",
                               root/"official_sources"/f"longbench_prompts_{commit}.json")
        prompts=json.loads(prompts_file.read_text())
        source_meta["longbench"]={"dataset_revision":revision,"prompt_revision":commit,"version":1}
        with zipfile.ZipFile(archive) as z:
            for task in LONG_TASKS:
                members=[name for name in z.namelist() if Path(name).name==task+".jsonl"]
                if len(members)!=1:raise RuntimeError(f"Cannot identify LongBench v1 task {task} in archive")
                records=[json.loads(line) for line in z.read(members[0]).decode().splitlines() if line.strip()]
                records.sort(key=lambda r:stable_seed(cfg.data_seed,r.get("_id",r.get("input",""))))
                if cfg.max_long_examples:records=records[:cfg.max_long_examples]
                for i,row in enumerate(records):
                    pid="longbench_"+task+"_"+str(row.get("_id",i))
                    for model,tok in tokenizers.items():
                        ids=middle_truncated_prompt(tok,prompts[task],row,max(cfg.contexts))
                        add(model,"LongBench",task,max(cfg.contexts),"evaluation",pid,ids,
                            [str(a) for a in row["answers"]],256 if task=="multi_news" else 128)
    if "ruler" in cfg.datasets:
        groups=prepare_ruler_records(cfg,tokenizers)
        source_meta["ruler"]={"version":1,"tasks":list(RULER_TASKS),"lengths_are_input_caps":True,
                              "same_raw_instances_across_models":True}
        for (context,split,task),records in groups.items():
            for i,row in enumerate(records):
                pid=f"ruler_{context}_{split}_{task}_{row.get('index',i)}"
                for model,tok in tokenizers.items():
                    ids=encode_chat(tok,row["input"],row.get("answer_prefix",""))
                    add(model,"RULER",task,context,split,pid,ids,[str(a) for a in row["outputs"]],128)
    # Identity, document and split checks are performed before any calibration.
    for model,rows in data.items():
        ids=[r["prompt_id"] for r in rows]
        if len(ids)!=len(set(ids)):raise RuntimeError("Duplicate prompt identity")
        cal={r["prompt_id"] for r in rows if r["split"]=="calibration"}
        eva={r["prompt_id"] for r in rows if r["split"]=="evaluation"}
        if cal & eva:raise RuntimeError("Calibration/evaluation overlap")
        if any(r["dataset"]=="LongBench" and r["split"]=="calibration" for r in rows):
            raise RuntimeError("LongBench test examples entered calibration")
        path=root/(slug(model)+".jsonl")
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(".tmp")
        with temp.open("w",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        temp.replace(path)
    atomic_json(done,{"signature":signature,"sources":source_meta,
                     "counts":{m:len(r) for m,r in data.items()}})
    return data


# ------------------- EXECUTION AND MATCHED-PREFIX AUDITS ----------------

def method_grid(cfg:Config) -> list[Method]:
    methods=[Method("dense")]
    methods.extend(Method("quest",fraction=x) for x in cfg.quest_fractions)
    methods.extend(Method("magicpig",bits=k,tables=l) for k,l in cfg.magic_settings)
    methods.append(Method("magicpig",dense_endpoint=True))
    methods.extend(Method("progressive",degree=1,stages=s) for s in cfg.stages)
    return methods


def eos_ids(runner:LastLayerRunner) -> set[int]:
    value=runner.model.generation_config.eos_token_id
    if value is None:value=runner.tokenizer.eos_token_id
    if value is None:return set()
    return set(value if isinstance(value,list) else [value])


def reference_continuation(runner:LastLayerRunner,example:dict,policy:Policy,cfg:Config) -> list[int]:
    folder=Path(cfg.out)/"reference"/slug(example["model"])/policy.key
    path=folder/(slug(example["prompt_id"])+".json")
    if path.exists():return json.loads(path.read_text())["tokens"]
    ids=example["input_ids"]
    runner.prepare(ids,Method("dense"),17,cfg.prefixes+1)
    rng=np.random.default_rng(stable_seed(17,example["prompt_id"],policy.key,"reference"))
    current=ids[-1];tokens=[]
    for _ in range(cfg.prefixes):
        out=runner.step(current,Method("dense"),policy,rng)
        current=out.token;tokens.append(current)
        if current in eos_ids(runner):break
    atomic_json(path,{"tokens":tokens,"seed":17,"policy":dataclasses.asdict(policy)})
    return tokens


def measure_example(runner:LastLayerRunner,example:dict,policy:Policy,method:Method,
                    seed:int,reference:list[int],cfg:Config,diagnostics:bool=True) -> list[dict]:
    nsteps=min(cfg.prefixes,len(reference))
    info=runner.prepare(example["input_ids"],method,stable_seed(seed,method.key,"hash"),nsteps+1)
    current=example["input_ids"][-1]
    rows=[]
    for j in range(nsteps):
        original_n=int(runner.state.get_seq_length())
        snapshot=runner.aux.snapshot() if runner.aux is not None else None
        timings=[];outputs=[]
        # Each repetition restores the identical earlier KV state and summary
        # endpoints. Randomness varies across repetitions, not with output timing.
        for repetition in range(cfg.warmups+cfg.repeats):
            runner.crop(original_n)
            if snapshot is not None:runner.aux.restore(snapshot)
            rng=np.random.default_rng(stable_seed(seed,example["prompt_id"],policy.key,method.key,j,repetition))
            sync(runner.device);start=time.perf_counter()
            output=runner.step(current,method,policy,rng)
            sync(runner.device);elapsed=(time.perf_counter()-start)*1000
            if repetition>=cfg.warmups:
                timings.append(elapsed)
                outputs.append({"early":output.early,"pre_support":output.pre_support,
                    "reads":output.source.distinct,"bytes":output.source.bytes,"stages":output.stages})
            del output
        # One separate instrumented repetition measures logical operand volume.
        runner.crop(original_n)
        if snapshot is not None:runner.aux.restore(snapshot)
        rng=np.random.default_rng(stable_seed(seed,example["prompt_id"],policy.key,method.key,j,"operand"))
        instrumented=runner.step(current,method,policy,rng,instrument=True)
        ctx=runner.last_state
        assert ctx is not None
        with torch.inference_mode():
            dense=runner.dense_logits(ctx)  # not exposed to sampling, outside timing
            audit=runner.audit(ctx,method,policy,dense)
        physical_count=ctx["k"].shape[1]*ctx["k"].shape[-2]
        observed=certified=None
        if diagnostics and method.kind=="progressive" and method.degree==1 and method.stages==max(cfg.stages):
            f=Frontier(runner.aux,ctx["q"][0,:,0],SourceAccess(ctx["k"][0],ctx["v"][0]),
                       float(ctx["kwargs"].get("scaling",runner.attention.scaling)))
            observed,certified=f.score_widths()
        sampled_reads=statistics.fmean(o["reads"] for o in outputs)
        sampled_bytes=statistics.fmean(o["bytes"] for o in outputs)
        row={"model":example["model"],"dataset":example["dataset"],"task":example["task"],
            "context":example["context"],"actual_context":int(ctx["k"].shape[-2]),
            "split":example["split"],"prompt_id":example["prompt_id"],"prefix_index":j,
            "seed":seed,"policy":policy.key,"policy_label":policy.label,
            "method":method.key,"method_label":method.label,"method_spec":dataclasses.asdict(method),
            "whole_ms":statistics.fmean(timings),"whole_ms_samples":timings,
            "sampled_early":statistics.fmean(float(o["early"]) for o in outputs),
            "sampled_pre_support":statistics.fmean(float(o["pre_support"]) for o in outputs),
            "sampled_reads":sampled_reads,"sampled_source_bytes":sampled_bytes,
            "sampled_stages":statistics.fmean(o["stages"] for o in outputs),
            "source_population":physical_count,
            "auxiliary_operand_bytes":instrumented.auxiliary_operand_bytes,
            "observed_widths":observed,"enclosure_widths":certified,
            "certificate_status":CERTIFICATE_STATUS,"hardware_bytes":None,
            "prefill_ms":info["prefill_ms"] if j==0 else None,
            "construction_ms":info["construction_ms"] if j==0 else None,
            "auxiliary_retained_bytes":runner.aux.storage_bytes() if runner.aux is not None else 0,
            **audit}
        if method.kind=="progressive" and not method.is_dense:
            row["early_percent"]=100*audit["expected_early"]
            row["pre_support_percent"]=100*audit["expected_pre_support"]
            row["completion_percent"]=100*(1-audit["expected_early"])
        elif method.is_dense:
            row["early_percent"]=row["pre_support_percent"]=0.0
            row["completion_percent"]=100.0
        else:
            row["early_percent"]=row["pre_support_percent"]=None
            row["completion_percent"]=None
        row["read_reduction_percent"]=100*(1-audit["expected_reads"]/physical_count)
        row["source_mib"]=audit["expected_source_bytes"]/MIB
        row["auxiliary_mib"]=instrumented.auxiliary_operand_bytes/MIB
        rows.append(row)
        # The instrumented run leaves all earlier and final-layer KV entries at
        # the common dense prefix. The next input is the DENSE path's token.
        current=reference[j]
        del outputs,instrumented,ctx
    return rows


def numeric_values(rows:list[dict],key:str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]


def nested_prompt_means(rows:list[dict],key:str) -> dict[tuple,list[float]]:
    prompt=collections.defaultdict(list)
    for row in rows:
        value=row.get(key)
        if value is not None and math.isfinite(float(value)):
            stratum=(row["dataset"],row["task"],row["context"])
            prompt[stratum+(row["prompt_id"],)].append(float(value))
    strata=collections.defaultdict(list)
    for k,values in prompt.items():strata[k[:3]].append(statistics.fmean(values))
    return dict(strata)


def equal_dataset_mean(strata:dict[tuple,list[float]]) -> float:
    by_dataset=collections.defaultdict(list)
    for (dataset,task,context),values in strata.items():
        if values:by_dataset[dataset].append(statistics.fmean(values))
    if not by_dataset:return math.nan
    return statistics.fmean(statistics.fmean(v) for v in by_dataset.values())


def balanced_mean(rows:list[dict],key:str) -> float:
    return equal_dataset_mean(nested_prompt_means(rows,key))


def bootstrap_ci(rows:list[dict],key:str,reps:int,seed:int=13) -> tuple[float,float,float]:
    strata=nested_prompt_means(rows,key)
    mean=equal_dataset_mean(strata)
    if not strata or reps<=0:return mean,math.nan,math.nan
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(reps):
        resampled={k:rng.choice(v,size=len(v),replace=True).tolist() for k,v in strata.items()}
        draws.append(equal_dataset_mean(resampled))
    low,high=np.quantile(draws,[.025,.975])
    return mean,float(low),float(high)


def weighted_quantile(values:Sequence[float],weights:Sequence[float],quantiles:Sequence[float]) -> list[float]:
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    keep=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    values,weights=values[keep],weights[keep]
    if not values.size:return [math.nan]*len(quantiles)
    order=np.argsort(values,kind="stable");values,weights=values[order],weights[order]
    cdf=np.cumsum(weights);cdf/=cdf[-1]
    ids=np.searchsorted(cdf,np.asarray(quantiles),side="left").clip(0,len(values)-1)
    return values[ids].tolist()


def weighted_rows(rows:list[dict],key:str,array:bool=False) -> tuple[list[float],list[float]]:
    # Equal dataset, stratum, prompt, prefix, seed, then block/head weights.
    tree=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    for r in rows:
        value=r.get(key)
        if value is None:continue
        tree[r["dataset"]][(r["task"],r["context"])][r["prompt_id"]].append(r)
    values=[];weights=[]
    for ds,strata in tree.items():
        for stratum,prompts in strata.items():
            for pid,items in prompts.items():
                for row in items:
                    vals=row[key] if array else [row[key]]
                    if not vals:continue
                    w=1/(len(tree)*len(strata)*len(prompts)*len(items)*len(vals))
                    values.extend(vals);weights.extend([w]*len(vals))
    return values,weights


def average_prefix_seeds(rows:list[dict],metric:str) -> list[dict]:
    """Average stochastic-seed measurements before quantiles over prefixes.

    A high-variance individual hash seed is not a distinct language-model prefix.
    Mean accuracy and prefix-tail accuracy use the same random-seed averaging.
    """
    groups=collections.defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            key=tuple(row.get(k) for k in ("model","policy","method","dataset","task",
                                          "context","prompt_id","prefix_index"))
            groups[key].append(row)
    return [{**group[0],metric:statistics.fmean(float(r[metric]) for r in group),
             "seed_count":len(group)} for group in groups.values()]


def choose_operating_points(rows:list[dict],policy:Policy,cfg:Config) -> tuple[list[Method],list[dict]]:
    relevant=[r for r in rows if r["split"]=="calibration" and r["policy"]==policy.key]
    if not relevant:raise RuntimeError("Calibration is empty. FineWeb or RULER calibration is required.")
    decisions=[];selected=[Method("dense")]
    for kind in ("quest","magicpig","progressive"):
        keys=sorted({r["method"] for r in relevant if r["method_spec"]["kind"]==kind})
        eligible=[]
        for key in keys:
            group=[r for r in relevant if r["method"]==key]
            mean=balanced_mean(group,"conditional_tv")
            prefix_means=average_prefix_seeds(group,"conditional_tv")
            values,weights=weighted_rows(prefix_means,"conditional_tv")
            p99=weighted_quantile(values,weights,[.99])[0]
            latency=balanced_mean(group,"whole_ms")
            valid=mean<=cfg.mean_tv and p99<=cfg.p99_tv
            decisions.append({"policy":policy.key,"method":key,"mean_tv":mean,"p99_tv":p99,
                              "whole_ms":latency,"meets_accuracy":valid,"selected":False})
            if valid:eligible.append((latency,key,Method(**group[0]["method_spec"])))
        if not eligible:raise RuntimeError(f"No calibrated {kind} setting meets the declared accuracy thresholds")
        _,key,method=min(eligible,key=lambda x:(x[0],x[1]))
        selected.append(method)
        for d in decisions:
            if d["method"]==key:d["selected"]=True
        if kind=="progressive":selected.append(dataclasses.replace(method,support_first=True))
    return selected,decisions


def normalize_answer(text:str) -> list[str]:
    text=text.lower().translate(str.maketrans("","",string.punctuation))
    text=re.sub(r"\b(a|an|the)\b"," ",text)
    return text.split()


def task_score(prediction:str,answers:list[str],task:str,dataset:str) -> float:
    if dataset=="RULER":
        # Official RULER retrieval scoring: percentage of expected strings found.
        return 100*statistics.fmean(float(a.lower() in prediction.lower()) for a in answers) if answers else 0.0
    if task=="multi_news":
        from rouge import Rouge
        if not prediction.strip():return 0.0
        rouge=Rouge()
        values=[]
        for a in answers:
            try:values.append(rouge.get_scores([prediction],[a],avg=True)["rouge-l"]["f"])
            except ValueError:values.append(0.0)
        return 100*max(values,default=0.0)
    p=normalize_answer(prediction)
    scores=[]
    for answer in answers:
        a=normalize_answer(answer)
        overlap=sum((collections.Counter(p)&collections.Counter(a)).values())
        scores.append(2*overlap/(len(p)+len(a)) if p and a else float(p==a))
    return 100*max(scores,default=0.0)


def generate_task(runner:LastLayerRunner,example:dict,method:Method,policy:Policy,seed:int,cfg:Config) -> dict:
    limit=example["max_new_tokens"]
    if cfg.suite=="smoke":limit=min(limit,16)
    info=runner.prepare(example["input_ids"],method,stable_seed(seed,method.key,"hash"),limit+1)
    current=example["input_ids"][-1];tokens=[];times=[];early=0;pre=0;source_bytes=0
    rng=np.random.default_rng(stable_seed(seed,example["prompt_id"],policy.key,method.key,"generation"))
    for _ in range(limit):
        sync(runner.device);start=time.perf_counter()
        out=runner.step(current,method,policy,rng)
        sync(runner.device);times.append((time.perf_counter()-start)*1000)
        # Audit the actually visited prefix separately. This catches distribution
        # drift at prefixes absent from the matched dense continuation.
        ctx=runner.last_state
        with torch.inference_mode():
            dense=runner.dense_logits(ctx)
            if method.kind=="progressive" and not method.is_dense:
                runner.audit(ctx,method,policy,dense)
        tokens.append(out.token);current=out.token
        early+=out.early;pre+=out.pre_support;source_bytes+=out.source.bytes
        if current in eos_ids(runner):break
    text=runner.tokenizer.decode(tokens,skip_special_tokens=True)
    return {"model":example["model"],"dataset":example["dataset"],"task":example["task"],
        "context":example["context"],"actual_context":example["actual_context"],
        "prompt_id":example["prompt_id"],"method":method.key,"method_label":method.label,
        "policy":policy.key,"seed":seed,"score":task_score(text,example["answers"],example["task"],example["dataset"]),
        "generated_tokens":tokens,"prediction":text,"output_tokens":len(tokens),
        "whole_ms":statistics.fmean(times),"first_token_ms":info["prefill_ms"]+info["construction_ms"]+times[0],
        "early_percent":100*early/len(tokens),"pre_support_percent":100*pre/len(tokens),
        "source_mib":source_bytes/len(tokens)/MIB,"construction_ms":info["construction_ms"],
        "certificate_status":CERTIFICATE_STATUS,"generation_limit":limit}


def teacher_nll(runner:LastLayerRunner,example:dict,method:Method,seed:int,cfg:Config) -> dict:
    targets=example.get("teacher_targets",[])
    if cfg.suite=="smoke":targets=targets[:4]
    # Progressive and support-first target the same pre-truncation dense logits.
    actual_method=Method("dense") if method.kind=="progressive" else method
    runner.prepare(example["input_ids"],actual_method,stable_seed(seed,method.key,"hash"),len(targets)+1)
    current=example["input_ids"][-1];nll=[]
    for j,target in enumerate(targets):
        out=runner.step(current,actual_method,Policy("full",1.0,1.0),np.random.default_rng(j))
        nll.append(float(-torch.log_softmax(out.logits.double(),0)[target]))
        current=target
    return {"model":example["model"],"dataset":"FineWeb","task":"continuation",
            "context":example["context"],"prompt_id":example["prompt_id"],"method":method.key,
            "seed":seed,"nll":statistics.fmean(nll),"tokens":len(nll),
            "nll_source":("dense_pretruncation_reference" if method.kind=="progressive"
                          else "implemented_pretruncation_logits")}


# -------------------------- RESULT EXPORT -----------------------------

def tex_escape(text:Any) -> str:
    replacements={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$",
                  "#":r"\#","_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}"}
    return "".join(replacements.get(c,c) for c in str(text))


def fmt(value:Any,digits:int=3) -> str:
    if value is None:return "NA"
    try:
        x=float(value)
        if not math.isfinite(x):return "NA"
        if 0<abs(x)<10**(-digits):return f"{x:.2e}"
        return f"{x:.{digits}f}"
    except (TypeError,ValueError):return tex_escape(value)


def write_table(path:Path,columns:str,headers:list[str],rows:list[list[str]],
                caption:str,label:str,wide:bool=True) -> str:
    env="table*" if wide else "table"
    body=[f"\\begin{{{env}}}[t]",r"\centering\small",r"\setlength{\tabcolsep}{3pt}",
          f"\\begin{{tabular}}{{{columns}}}",r"\toprule"," & ".join(headers)+r" \\",r"\midrule"]
    body += [" & ".join(row)+r" \\" for row in rows]
    body += [r"\bottomrule",r"\end{tabular}","\\caption{"+caption+"}",
             "\\label{"+label+"}",f"\\end{{{env}}}",""]
    text="\n".join(body)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    return text


def write_csv(path:Path,rows:list[dict],fields:Optional[list[str]]=None) -> None:
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:(json.dumps(clean_json(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v)
                             for k,v in row.items()})


def summarize_numerical(rows:list[dict]) -> dict:
    def mx(k):return max(numeric_values(rows,k),default=math.nan)
    return {"logit_violations":sum(numeric_values(rows,"logit_violations")),
            "certificate_excess":mx("certificate_excess"),
            "normalization_error":mx("normalization_error"),
            "support_violations":sum(numeric_values(rows,"support_violations")),
            "conditional_tv_max":mx("conditional_tv")}


def aggregate(cfg:Config,fill_tex:Optional[str]=None) -> None:
    root=Path(cfg.out)
    rows=jsonl_read(root/"matched.jsonl")
    if not rows:raise RuntimeError("No matched-prefix measurements were found")
    selection_file=root/"selected_methods.json"
    selected=json.loads(selection_file.read_text()) if selection_file.exists() else {}
    evaluation=[r for r in rows if r["split"]=="evaluation"]
    tables=root/"tables";tables.mkdir(parents=True,exist_ok=True)
    confidence=[];main=[];summary_rows=[]
    hw=jsonl_read(root/"hardware.jsonl")
    hw_dense={(r["model"],r["policy"],r["prompt_id"],r["prefix_index"],r["seed"]):r
              for r in hw if r["method"]=="dense"}
    for row in hw:
        dense=hw_dense.get((row["model"],row["policy"],row["prompt_id"],row["prefix_index"],row["seed"]))
        if dense and dense.get("device_bytes",0)>0:
            row["traffic_ratio"]=row["device_bytes"]/dense["device_bytes"]
    replacements={}
    for model in cfg.models:
        model_name="Qwen3" if "Qwen" in model else "Llama"
        for pkey,methods in selected.get(model,{}).items():
            ordered=sorted((Method(**m) for m in methods),key=lambda m:(
                {"dense":0,"quest":1,"magicpig":2,"progressive":4}[m.kind]-(1 if m.support_first else 0)))
            for method in ordered:
                group=[r for r in evaluation if r["model"]==model and r["policy"]==pkey and r["method"]==method.key]
                if not group:continue
                means={key:balanced_mean(group,key) for key in ("early_percent","pre_support_percent",
                    "read_reduction_percent","source_mib","auxiliary_mib","whole_ms","conditional_tv")}
                counters=[r for r in hw if r["model"]==model and r["policy"]==pkey and r["method"]==method.key
                          and r.get("traffic_ratio") is not None]
                ratio=balanced_mean(counters,"traffic_ratio") if counters else math.nan
                if method.kind=="dense":ratio=1.0 # normalization identity, not a measured byte count
                means["traffic_ratio"]=ratio
                main.append([model_name,tex_escape(group[0]["policy_label"]),tex_escape(method.label)]+
                            [fmt(means[k]) for k in ("early_percent","pre_support_percent","read_reduction_percent",
                                "source_mib","auxiliary_mib","traffic_ratio","whole_ms","conditional_tv")])
                summary_rows.append({"model":model,"policy":pkey,"method":method.key,
                    "prompt_count":len({r["prompt_id"] for r in group}),
                    "prefix_seed_count":len(group),"counter_prefix_seed_count":len(counters),**means})
                for key in means:
                    if key=="traffic_ratio":continue
                    mean,lower,upper=bootstrap_ci(group,key,cfg.bootstrap)
                    confidence.append({"model":model,"policy":pkey,"method":method.key,"metric":key,
                                       "mean":mean,"ci_low":lower,"ci_high":upper})
    caption=("Final-layer comparison on the measured prompts. Source bytes count original-record requests, "
        "and auxiliary bytes count logical tensor operands, including interval and vocabulary work. "
        "The whole-decoder traffic ratio uses Nsight Compute device counters on the separately recorded profiling subset. "
        "NA indicates an uncollected counter. Latency measures this PyTorch implementation, including maintenance and sampling. "
        "Progressive certificates are numerically validated. Dataset means receive equal weight within each model.")
    replacements["tab:opentraffic"]=write_table(tables/"main_emission_traffic.tex","lllrrrrrrrr",
        ["Model","Policy","Method","Early (\\%)","Pre-support (\\%)","Read reduction (\\%)",
         "Source MiB","Auxiliary MiB","HBM ratio","ms/token","TV"],main,caption,"tab:opentraffic")
    width_rows=[];width_detail=[]
    for model in cfg.models:
        for ds in ("FineWeb","LongBench","RULER"):
            group=[r for r in evaluation if r["model"]==model and r["dataset"]==ds and r.get("observed_widths")]
            if not group:continue
            values,weights=weighted_rows(group,"observed_widths",True)
            obs=weighted_quantile(values,weights,[.5,.9,.99])
            values,weights=weighted_rows(group,"enclosure_widths",True)
            enc=weighted_quantile(values,weights,[.5,.9,.99])
            width_rows.append([tex_escape("Qwen3" if "Qwen" in model else "Llama"),ds,
                               " / ".join(map(fmt,obs))," / ".join(map(fmt,enc))])
            for context in sorted({r["context"] for r in group}):
                gg=[r for r in group if r["context"]==context]
                o=weighted_quantile(*weighted_rows(gg,"observed_widths",True),[.5,.9,.99])
                e=weighted_quantile(*weighted_rows(gg,"enclosure_widths",True),[.5,.9,.99])
                width_detail.append({"model":model,"dataset":ds,"context":context,
                    "observed_median":o[0],"observed_p90":o[1],"observed_p99":o[2],
                    "enclosure_median":e[0],"enclosure_p90":e[1],"enclosure_p99":e[2]})
    replacements["tab:openwidth"]=write_table(tables/"main_score_widths.tex","llcc",
        ["Model","Data",r"$\widehat\omega$",r"$\omega$"],width_rows,
        f"Actual initial-frontier score widths for blocks of at most {cfg.initial} records. "
        "Entries give the median, 90th percentile, and 99th percentile. Prompts, prefix positions, and head-block pairs use nested equal weights.",
        "tab:openwidth",False)
    grid_rows=[]
    for model in cfg.models:
        for pkey in sorted({r["policy"] for r in evaluation if r["model"]==model}):
            for key in sorted({r["method"] for r in evaluation if r["model"]==model and r["policy"]==pkey}):
                group=[r for r in evaluation if r["model"]==model and r["policy"]==pkey and r["method"]==key]
                grid_rows.append({"model":model,"policy":pkey,"method":key,
                    **{k:balanced_mean(group,k) for k in ("conditional_tv","whole_ms","early_percent",
                    "pre_support_percent","read_reduction_percent","completion_percent","expected_stages")}})
    write_csv(root/"operating_points.csv",grid_rows)
    write_table(tables/"operating_points.tex","lllrrrr",
        ["Model","Policy","Setting","TV","ms/token","Early (\\%)","Read reduction (\\%)"],
        [[tex_escape(r[k]) for k in ("model","policy","method")]+[fmt(r[k]) for k in
          ("conditional_tv","whole_ms","early_percent","read_reduction_percent")] for r in grid_rows],
        "All evaluated operating points, including dense endpoints. Calibration and evaluation are disjoint.","tab:implementedgrid")
    numerical=[];stage_rows=[]
    for model in cfg.models:
        for pkey in sorted({r["policy"] for r in evaluation if r["model"]==model}):
            group=[r for r in evaluation if r["model"]==model and r["policy"]==pkey
                   and r["method_spec"]["kind"]=="progressive" and not r["method_spec"]["support_first"]]
            if not group:continue
            numerical.append({"model":model,"policy":pkey,**summarize_numerical(group)})
            diagnostic=[r for r in group if r["method_spec"]["degree"]==1 and r["method_spec"]["stages"]==max(cfg.stages)]
            for stage in range(max(cfg.stages)):
                expanded=[]
                for row in diagnostic:
                    if not row["audit_stages"]:continue
                    st=row["audit_stages"][min(stage,len(row["audit_stages"])-1)]
                    expanded.append({**row,**st})
                if expanded:
                    stage_rows.append({"model":model,"policy":pkey,"stage":stage,
                        **{k:balanced_mean(expanded,k) for k in ("attention_radius","logit_width","C","source_bytes")}})
    write_csv(root/"numerical_checks.csv",numerical)
    write_csv(root/"stage_diagnostics.csv",stage_rows)
    replacements["tab:opennumerical"]=write_table(tables/"numerical_checks.tex","llrrrrr",
        ["Model","Policy","Logit violations","Certificate excess","Normalization error","Support violations","Law TV"],
        [[tex_escape(r["model"]),tex_escape(r["policy"])]+[fmt(r[k]) for k in
          ("logit_violations","certificate_excess","normalization_error","support_violations","conditional_tv_max")]
          for r in numerical],
        "Full-vocabulary numerical audits of progressive emission and conditional completion. "
        "These checks validate the implemented numerical runs and are distinct from a verified-arithmetic proof.","tab:opennumerical")
    write_table(tables/"stage_diagnostics.tex","llrrrrr",
        ["Model","Policy","Stage","Attention radius","Logit width",r"$C_r$","Source bytes"],
        [[tex_escape(r["model"]),tex_escape(r["policy"]),str(r["stage"])]+[fmt(r[k]) for k in
         ("attention_radius","logit_width","C","source_bytes")] for r in stage_rows],
        "Numerical certificates along the fully deferred path. Source accounting excludes dense audit reads.","tab:implementedstages")
    ablation=[r for r in grid_rows if re.match(r"progressive_d[01]_s2$",r["method"])]
    write_table(tables/"summary_degree.tex","lllrrrrr",
        ["Model","Policy","Setting","Early (\\%)","Pre-support (\\%)","Read reduction (\\%)","Completion (\\%)","ms/token"],
        [[tex_escape(r[k]) for k in ("model","policy","method")]+[fmt(r[k]) for k in
         ("early_percent","pre_support_percent","read_reduction_percent","completion_percent","whole_ms")] for r in ablation],
        "Summary-degree comparison at the same two-stage cap.","tab:openablation")
    tasks=jsonl_read(root/"task_outputs.jsonl")
    quality=[]
    for key,items in itertools.groupby(sorted(tasks,key=lambda r:(r["model"],r["policy"],r["method"],r["dataset"],r["task"])),
          key=lambda r:(r["model"],r["policy"],r["method"],r["dataset"],r["task"])):
        group=list(items)
        score,low,high=bootstrap_ci(group,"score",cfg.bootstrap)
        quality.append(dict(zip(("model","policy","method","dataset","task"),key))|
                       {"score":score,"ci_low":low,"ci_high":high,"whole_ms":balanced_mean(group,"whole_ms")})
    write_csv(root/"task_quality.csv",quality)
    write_table(tables/"task_quality.tex","llllrr",
        ["Model","Policy","Method","Task","Score","ms/token"],
        [[tex_escape(r[k]) for k in ("model","policy","method","task")]+[fmt(r["score"]),fmt(r["whole_ms"])] for r in quality],
        "Each method generates its own continuation. Scores retain the task-level breakdown.","tab:implementedquality")
    write_csv(root/"main_results.csv",summary_rows)
    write_csv(root/"confidence_intervals.csv",confidence)
    write_csv(root/"score_widths.csv",width_detail)
    write_csv(root/"hardware_coverage.csv",hw)
    write_csv(root/"nll.csv",jsonl_read(root/"nll.jsonl"))
    atomic_json(root/"coverage.json",{"models":cfg.models,"requested_datasets":cfg.datasets,
        "actual_datasets":sorted({r["dataset"] for r in evaluation}),"evaluation_rows":len(evaluation),
        "hardware_rows":len(hw),"certificate_status":CERTIFICATE_STATUS,
        "unimplemented_manuscript_experiments":["host-resident placement","all-layer baseline intervention",
              "batch-four throughput","formal random-bit arithmetic","sensitivity settings outside the configured run"],
        "unmeasured_cells_are_NA":True})
    if fill_tex:
        original=Path(fill_tex).read_text(encoding="utf-8")
        changed=[]
        def replace_table(match:re.Match) -> str:
            text=match.group(0)
            label=re.search(r"\\label\{([^}]+)\}",text)
            if label and label.group(1) in replacements:
                changed.append(label.group(1));return replacements[label.group(1)]
            return text
        filled=re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",replace_table,original,flags=re.S)
        target=root/(Path(fill_tex).stem+".measured.tex")
        target.write_text(filled,encoding="utf-8")
        atomic_json(root/"filled_tables.json",{"source":fill_tex,"output":str(target),"replaced_labels":changed,
            "pagination_not_verified":True})
    LOG.info("Wrote CSV summaries and LaTeX tables to %s",root.resolve())


# -------------------------- GPU COUNTER RUNS ---------------------------

def parse_ncu_csv(path:Path) -> dict:
    text=path.read_text(encoding="utf-8",errors="replace")
    lines=text.splitlines()
    start=next((i for i,s in enumerate(lines) if '"Metric Name"' in s and '"Metric Value"' in s),None)
    if start is None:raise RuntimeError(f"No Nsight metric table in {path}. Check GPU performance-counter permission.")
    reader=csv.DictReader(io.StringIO("\n".join(lines[start:])))
    totals={"dram__bytes_read.sum":0.0,"dram__bytes_write.sum":0.0}
    seen=set();found=set()
    units={"byte":1.0,"bytes":1.0,"kbyte":1000.0,"mbyte":1e6,"gbyte":1e9,"kib":1024.0,"mib":MIB}
    for row in reader:
        metric=row.get("Metric Name","")
        if metric not in totals:continue
        key=(row.get("Process ID"),row.get("ID"),row.get("Kernel Name"),metric)
        if key in seen:continue
        seen.add(key)
        value=str(row.get("Metric Value","")).replace(",","").strip()
        try:number=float(value)
        except ValueError as exc:raise RuntimeError(f"Unavailable hardware metric {metric}: {value}") from exc
        unit=str(row.get("Metric Unit","byte")).strip().lower()
        if unit not in units:raise RuntimeError(f"Unrecognized counter unit {unit}")
        totals[metric]+=number*units[unit];found.add(metric)
    if found!=set(totals):raise RuntimeError("Both device read and write counters are required")
    return {"device_read_bytes":totals["dram__bytes_read.sum"],
            "device_write_bytes":totals["dram__bytes_write.sum"],
            "device_bytes":sum(totals.values()),"kernel_metric_rows":len(seen)}


def counter_worker(job_path:Path) -> None:
    job=json.loads(job_path.read_text())
    cfg=Config(**job["config"]);cfg.counter_worker=True
    cfg.revision=job["revision"]
    model,tok,_=load_checkpoint(job["example"]["model"],cfg)
    runner=LastLayerRunner(model,tok,cfg)
    method=Method(**job["method"]);policy=Policy(**job["policy"])
    ex=job["example"];seed=job["seed"];target=job["prefix_index"]
    runner.prepare(ex["input_ids"],method,stable_seed(seed,method.key,"hash"),target+2)
    current=ex["input_ids"][-1]
    for j in range(target):
        runner.step(current,method,policy,np.random.default_rng(stable_seed(seed,j,"advance")))
        current=job["reference"][j]
    length=int(runner.state.get_seq_length())
    snapshot=runner.aux.snapshot() if runner.aux is not None else None
    for repetition in range(cfg.warmups):
        runner.crop(length)
        if snapshot is not None:runner.aux.restore(snapshot)
        runner.step(current,method,policy,np.random.default_rng(stable_seed(seed,target,repetition,"warm")))
    runner.crop(length)
    if snapshot is not None:runner.aux.restore(snapshot)
    rng=np.random.default_rng(stable_seed(seed,ex["prompt_id"],policy.key,method.key,target,cfg.warmups))
    sync(runner.device)
    torch.cuda.nvtx.range_push("REA_TOKEN")
    result=runner.step(current,method,policy,rng)
    sync(runner.device)
    torch.cuda.nvtx.range_pop()
    atomic_json(job_path.with_suffix(".worker.json"),{"token":result.token,"early":result.early,
        "logical_source_bytes":result.source.bytes,"profiled_tokens":1})
    runner.close()


def profile_hardware(cfg:Config) -> None:
    executable=shutil.which(cfg.ncu)
    if executable is None:
        raise RuntimeError("Nsight Compute CLI (ncu) is required for actual HBM transactions. No byte estimate was substituted.")
    root=Path(cfg.out)
    selection=json.loads((root/"selected_methods.json").read_text())
    existing=jsonl_read(root/"hardware.jsonl")
    done={(r["model"],r["policy"],r["method"],r["prompt_id"],r["prefix_index"],r["seed"]) for r in existing}
    script=Path(__file__).resolve()
    for model_id in cfg.models:
        examples=jsonl_read(root/"data"/(slug(model_id)+".jsonl"))
        strata=collections.defaultdict(list)
        for e in examples:
            if e["split"]=="evaluation":strata[e["dataset"],e["task"],e["context"]].append(e)
        meta=json.loads((root/"metadata"/(slug(model_id)+".json")).read_text())
        for pkey,method_dicts in selection[model_id].items():
            policy=(Policy("topk",cfg.top_k,cfg.temperature) if pkey.startswith("topk_")
                    else Policy("nucleus",cfg.top_p,cfg.temperature))
            for stratum,group in strata.items():
                chosen=sorted(group,key=lambda e:stable_seed(cfg.data_seed,e["prompt_id"]))
                if cfg.profile_prompts>0:chosen=chosen[:cfg.profile_prompts]
                for ex in chosen:
                    ref_path=root/"reference"/slug(model_id)/pkey/(slug(ex["prompt_id"])+".json")
                    reference=json.loads(ref_path.read_text())["tokens"]
                    for j in range(min(cfg.profile_prefixes,len(reference))):
                        for md in method_dicts:
                            method=Method(**md)
                            for seed in cfg.seeds:
                                key=(model_id,pkey,method.key,ex["prompt_id"],j,seed)
                                if key in done:continue
                                ident=hashlib.sha256(repr(key).encode()).hexdigest()[:20]
                                job_path=root/"hardware_jobs"/(ident+".json")
                                csv_path=job_path.with_suffix(".ncu.csv")
                                atomic_json(job_path,{"config":dataclasses.asdict(cfg),"revision":meta["resolved_revision"],
                                    "example":ex,"method":md,"policy":dataclasses.asdict(policy),"seed":seed,
                                    "prefix_index":j,"reference":reference})
                                command=[executable,"--target-processes","all","--nvtx","--nvtx-include","REA_TOKEN/",
                                    "--metrics","dram__bytes_read.sum,dram__bytes_write.sum","--csv","--page","raw",
                                    "--cache-control","none","--clock-control","none","--log-file",str(csv_path.resolve()),
                                    sys.executable,str(script),"--counter-worker",str(job_path.resolve())]
                                LOG.info("Profiling HBM: %s %s %s prefix=%d",model_id,method.key,ex["prompt_id"],j)
                                log_path=job_path.with_suffix(".process.log")
                                with log_path.open("w") as output:
                                    subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)
                                counters=parse_ncu_csv(csv_path)
                                append_jsonl(root/"hardware.jsonl",{"model":model_id,"policy":pkey,"method":method.key,
                                    "dataset":ex["dataset"],"task":ex["task"],"context":ex["context"],
                                    "prompt_id":ex["prompt_id"],"prefix_index":j,"seed":seed,
                                    "counter_source":"Nsight Compute","profiled_tokens":1,**counters})
                                done.add(key)
    aggregate(cfg)


# ----------------------------- RUN DRIVER -----------------------------

def config_identity(cfg:Config) -> str:
    ignored={"overwrite","profile_prompts","profile_prefixes","ncu","counter_worker","prepare_ruler"}
    content={k:v for k,v in dataclasses.asdict(cfg).items() if k not in ignored}
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def run(cfg:Config,fill_tex:Optional[str]=None) -> None:
    root=Path(cfg.out);root.mkdir(parents=True,exist_ok=True)
    config_path=root/"config.json"
    if config_path.exists():
        saved=Config(**json.loads(config_path.read_text()))
        if config_identity(saved)!=config_identity(cfg):
            raise RuntimeError("This output directory belongs to a different experiment. Use another --out directory.")
    else:atomic_json(config_path,dataclasses.asdict(cfg))
    examples_by_model=prepare_data(cfg)
    all_rows=jsonl_read(root/"matched.jsonl")
    # A completed prompt marker is written only after every measured prefix and
    # audit is saved. Partial prompt rows are removed before a resumed attempt.
    completed=jsonl_read(root/"completed.jsonl")
    done={tuple(x["key"]) for x in completed}
    def unit_key(model,policy,method,seed,ex):return model,policy.key,method.key,seed,ex["prompt_id"]
    def process(runner,ex,policy,method,seed,ref):
        key=unit_key(ex["model"],policy,method,seed,ex)
        if key in done:return
        LOG.info("%s | %s | %s | %s | seed=%d",ex["split"],ex["model"],ex["prompt_id"],method.key,seed)
        result=measure_example(runner,ex,policy,method,seed,ref,cfg,diagnostics=seed==cfg.seeds[0])
        for row in result:append_jsonl(root/"matched.jsonl",row)
        all_rows.extend(result)
        append_jsonl(root/"completed.jsonl",{"key":key,"rows":len(result)})
        done.add(key)
    # Detect incomplete writes; never average duplicate partial measurements.
    filtered=[]
    for row in all_rows:
        key=(row["model"],row["policy"],row["method"],row["seed"],row["prompt_id"])
        if key in done:filtered.append(row)
    if len(filtered)!=len(all_rows):
        tmp=root/"matched.recovered.jsonl"
        with tmp.open("w",encoding="utf-8") as f:
            for row in filtered:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        tmp.replace(root/"matched.jsonl");all_rows=filtered
    selection_path=root/"selected_methods.json"
    selected=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    calibration_decisions=[]
    tasks_done={(r["model"],r["policy"],r["method"],r["seed"],r["prompt_id"])
                for r in jsonl_read(root/"task_outputs.jsonl")}
    nll_done={(r["model"],r["method"],r["seed"],r["prompt_id"])
              for r in jsonl_read(root/"nll.jsonl")}
    for model_id in cfg.models:
        LOG.info("Loading %s",model_id)
        model,tok,meta=load_checkpoint(model_id,cfg)
        metadata_path=root/"metadata"/(slug(model_id)+".json")
        if metadata_path.exists():
            previous=json.loads(metadata_path.read_text())
            if previous.get("resolved_revision")!=meta.get("resolved_revision"):
                raise RuntimeError("Checkpoint revision changed during a resumed experiment")
        atomic_json(metadata_path,meta)
        runner=validate_native_adapter(model,tok,cfg)
        policies=[Policy("topk",cfg.top_k,cfg.temperature),Policy("nucleus",cfg.top_p,cfg.temperature)]
        examples=examples_by_model[model_id]
        try:
            for policy in policies:
                grid=method_grid(cfg)
                calibration=[e for e in examples if e["split"]=="calibration"]
                for ex in calibration:
                    ref=reference_continuation(runner,ex,policy,cfg)
                    # Alternate deterministic method orders across prompts.
                    ordered=grid if stable_seed(ex["prompt_id"])%2 else grid[::-1]
                    for method in ordered:
                        for seed in cfg.seeds:process(runner,ex,policy,method,seed,ref)
                best,decisions=choose_operating_points([r for r in all_rows if r["model"]==model_id],policy,cfg)
                for d in decisions:d["model"]=model_id
                calibration_decisions+=decisions
                selected.setdefault(model_id,{})[policy.key]=[dataclasses.asdict(m) for m in best]
                atomic_json(selection_path,selected)
                evaluation=[e for e in examples if e["split"]=="evaluation"]
                methods={m.key:m for m in (grid if cfg.eval_grid else best)}
                methods.update({m.key:m for m in best})
                # Preserve a four-stage geometry diagnostic and the paired degree
                # ablation even when the selected stage cap is one.
                diagnostics=[Method("progressive",degree=1,stages=max(cfg.stages)),
                             Method("progressive",degree=0,stages=2),Method("progressive",degree=1,stages=2)]
                methods.update({m.key:m for m in diagnostics})
                for ex in evaluation:
                    ref=reference_continuation(runner,ex,policy,cfg)
                    ordered=list(methods.values())
                    if stable_seed(ex["prompt_id"])%2:ordered=ordered[::-1]
                    for method in ordered:
                        for seed in cfg.seeds:process(runner,ex,policy,method,seed,ref)
                    if cfg.generate_tasks:
                        for method in best:
                            for seed in cfg.seeds:
                                key=(model_id,policy.key,method.key,seed,ex["prompt_id"])
                                if ex["dataset"]!="FineWeb" and key not in tasks_done:
                                    task=generate_task(runner,ex,method,policy,seed,cfg)
                                    append_jsonl(root/"task_outputs.jsonl",task);tasks_done.add(key)
                                elif ex["dataset"]=="FineWeb":
                                    nk=(model_id,method.key,seed,ex["prompt_id"])
                                    if nk not in nll_done:
                                        row=teacher_nll(runner,ex,method,seed,cfg)
                                        append_jsonl(root/"nll.jsonl",row);nll_done.add(nk)
                aggregate(cfg,fill_tex)
        except Exception as exc:
            atomic_json(root/"last_failure.json",{"model":model_id,"error":str(exc),
                        "traceback":traceback.format_exc(),"failing_prompt_not_exported":True})
            raise
        finally:
            runner.close();del runner,model,tok;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    write_csv(root/"calibration.csv",calibration_decisions)
    aggregate(cfg,fill_tex)


# ----------------------------- SELF TESTS ------------------------------

def self_test() -> dict:
    """CPU-only tests. Synthetic fixtures are NEVER added to measurement files."""
    import tempfile
    from types import SimpleNamespace
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    torch.manual_seed(913)
    checks=collections.Counter()
    def require(condition:bool,name:str) -> None:
        if not condition:raise AssertionError(name)
        checks[name]+=1
    try:
        for _ in range(40):
            a=torch.randn(7,dtype=torch.float64)
            b=a+torch.rand(7,dtype=torch.float64)
            c=torch.randn(7,dtype=torch.float64)
            d=c+torch.rand(7,dtype=torch.float64)
            product=Box(a,b)*Box(c,d)
            for u,v in itertools.product((a,b),(c,d)):
                y=u*v
                require(bool(((y>=product.lo)&(y<=product.hi)).all()),"interval_product")
            norm_weight=torch.randn(7,dtype=torch.float64)
            nb=rms_box(Box(a[None],b[None]),norm_weight,1e-5,torch.float64)
            sb=silu_box(Box(a,b),torch.float64)
            for t in (0,.25,.5,.75,1):
                x=a+t*(b-a)
                nx=x*torch.rsqrt((x*x).mean()+1e-5)*norm_weight
                sx=F.silu(x)
                require(bool(((nx>=nb.lo[0])&(nx<=nb.hi[0])).all()),"rms_interval")
                require(bool(((sx>=sb.lo)&(sx<=sb.hi)).all()),"silu_interval")
        # Includes tie cases, singleton intervals, top-1, and nucleus boundary cases.
        policies=[Policy("topk",1,1),Policy("topk",2,1),Policy("topk",4,1),
                  Policy("nucleus",.8,1),Policy("nucleus",.9,1)]
        boxes=[]
        for _ in range(40):
            l=torch.randn(4,dtype=torch.float64)*2
            boxes.append(Box(l,l+torch.rand(4,dtype=torch.float64)*2))
        boxes += [Box.point(torch.tensor([1.,1.,0.,0.],dtype=torch.float64)),
                  Box(torch.tensor([3.,1.,-2.,-3.],dtype=torch.float64),
                      torch.tensor([3.01,1.01,-1.99,-2.99],dtype=torch.float64))]
        for box in boxes:
            for pol in policies:
                cert=probability_certificate(box,pol)
                for corner in itertools.product((0,1),repeat=4):
                    mix=torch.tensor(corner,dtype=torch.bool)
                    z=torch.where(mix,box.hi,box.lo)
                    p=policy_prob(z,pol)
                    require(bool((cert.lower<=p+1e-11).all()),"policy_certificate_corner")
                    require(not bool((cert.core&(p==0)).any()),"guaranteed_core")
        for degree in (0,1):
            k=(torch.randn(2,19,4)*.3).to(torch.bfloat16)
            v=(torch.randn(2,19,4)*.2).to(torch.bfloat16)
            q=(torch.randn(4,4)*.2).to(torch.bfloat16)
            tree=SummaryTree(k[:,:16],v[:,:16],4,8,degree,32)
            snapshot=tree.snapshot()
            tree.append(k[:,16],v[:,16]);tree.restore(snapshot)
            require(tree.n==16,"tree_rollback")
            for idx in range(16,19):tree.append(k[:,idx],v[:,idx])
            require(tree.n==19,"tree_append")
            exact=torch.softmax(torch.einsum("hd,hnd->hn",q.double()/2,k.repeat_interleave(2,0).double()),-1)
            exact=torch.einsum(
                "hn,hnd->hd",exact,v.repeat_interleave(2,0).double())
            source=SourceAccess(k,v)
            frontier=Frontier(tree,q,source,.5)
            for _ in range(30):
                enclosure=frontier.enclosure(torch.bfloat16)
                require(bool(((exact>=enclosure.lo)&(exact<=enclosure.hi)).all()),"attention_enclosure")
                if not frontier.refine():break
            require(source.distinct==2*19,"physical_record_union")
            observed,certified=frontier.score_widths()
            require(all(x<=y+1e-12 for x,y in zip(observed,certified)),"observed_vs_enclosure_width")
            require(tree.storage_bytes()>0,"summary_storage")
        # Check the inverted index against a direct scan of hash codes, including
        # pending appends and rollback. The scan is used only in this unit test.
        k=torch.randn(2,100,4)
        v=torch.randn_like(k)
        index=LSHIndex(k[:,:97],bits=3,tables=8,seed=11,append_limit=4)
        snap=index.snapshot();index.append(k[:,97],v[:,97]);index.restore(snap)
        require(index.n==97,"hash_rollback")
        for i in range(97,100):index.append(k[:,i],v[:,i])
        for g in range(2):
            query=torch.randn(4)
            mask=index.lookup(g,query)
            code=index.encode(query)
            all_codes=index.encode(k[g,4:]-index.center[g])
            truth=torch.zeros(100,dtype=torch.bool)
            truth[4:]=(all_codes==code).sum(-1)>=2
            truth[:4]=True;truth[-64:]=True
            require(bool(torch.equal(mask,truth)),"inverted_index_vs_scan")
        source=SourceAccess(k,v)
        oo=index.output(torch.randn(4,4),source,.5)
        require(bool(torch.isfinite(oo).all()),"magicpig_output")
        for a in (.05,.25,.5,.75,.95):
            # Analytic inclusion against explicit Binomial probability.
            cosine=torch.tensor([math.cos(math.pi*(1-a))],dtype=torch.float64)
            prob=float(LSHIndex.inclusion(cosine,3,8))
            success=a**3
            brute=sum(math.comb(8,j)*success**j*(1-success)**(8-j) for j in range(2,9))
            require(abs(prob-brute)<1e-12,"lsh_inclusion_probability")
        pages=PageIndex(k[:,:96],16)
        for i in range(96,100):pages.append(k[:,i],v[:,i])
        psource=SourceAccess(k,v)
        q=torch.randn(4,4)
        output=pages.output(q,psource,.5,1.0)
        exact=torch.einsum("hn,hnd->hd",torch.softmax(
            torch.einsum("hd,hnd->hn",q.double(),k.repeat_interleave(2,0).double())*.5,-1),
            v.repeat_interleave(2,0).double()).float()
        require(float((output-exact).abs().max())<1e-6,"quest_dense_endpoint")
        # Real suffix operators on a tiny random fixture, without Transformers.
        class TinyNorm(torch.nn.Module):
            def __init__(self,dim):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(dim,dtype=torch.float64));self.variance_epsilon=1e-5
            def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.variance_epsilon)*self.weight
        class TinyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__();self.gate_proj=torch.nn.Linear(8,12,bias=False,dtype=torch.float64)
                self.up_proj=torch.nn.Linear(8,12,bias=False,dtype=torch.float64)
                self.down_proj=torch.nn.Linear(12,8,bias=False,dtype=torch.float64)
            def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
        cfg=Config(device="cpu",dtype="float32",interval_chunk=16,stages=(1,2,4))
        r=object.__new__(LastLayerRunner)
        r.cfg=cfg;r.dtype=torch.float64;r.device=torch.device("cpu")
        r.attention=SimpleNamespace(o_proj=torch.nn.Linear(16,8,bias=False,dtype=torch.float64),scaling=.5)
        r.block=SimpleNamespace(mlp=TinyMLP(),post_attention_layernorm=TinyNorm(8))
        r.model=SimpleNamespace(config=SimpleNamespace(vocab_size=8,rms_norm_eps=1e-5),
                  model=SimpleNamespace(norm=TinyNorm(8)),lm_head=torch.nn.Linear(8,8,dtype=torch.float64))
        def native(module,q,k,v,mask,**kwargs):
            kk=k.repeat_interleave(2,1);vv=v.repeat_interleave(2,1)
            o=torch.softmax((q@kk.transpose(-1,-2))*.5,-1)@vv
            return o.transpose(1,2),None
        r.native_attention=native
        k=torch.randn(2,12,4,dtype=torch.float64)*.03
        v=torch.randn(2,12,4,dtype=torch.float64)*.01
        q=torch.randn(4,4,dtype=torch.float64)*.05
        h=torch.ones(1,1,8,dtype=torch.float64)+torch.randn(1,1,8,dtype=torch.float64)*.01
        ctx={"module":None,"q":q[None,:,None],"k":k[None],"v":v[None],"mask":None,"kwargs":{"scaling":.5},"h":h}
        with torch.inference_mode():
            for degree in (0,1):
                r.aux=SummaryTree(k,v,4,8,degree,32)
                reference=r.dense_logits(ctx)
                for pol in (Policy("topk",3,.7),Policy("nucleus",.9,.7)):
                    early=r.audit(ctx,Method("progressive",degree=degree,stages=4),pol,reference)
                    support=r.audit(ctx,Method("progressive",degree=degree,stages=4,support_first=True),pol,reference)
                    require(early["conditional_tv"]<1e-9,"conditional_completion_law")
                    require(early["expected_reads"]<=support["expected_reads"]+1e-9,"early_vs_support_reads")
                    require(support["expected_pre_support"]==0,"support_first_emission")
        # Independently test the probabilistic telescoping law and source cost.
        p=torch.tensor([.5,.3,.2],dtype=torch.float64)
        allocations=[torch.tensor([.1,.05,0.],dtype=torch.float64),torch.tensor([.3,.2,.1],dtype=torch.float64)]
        reconstructed=allocations[0]+(allocations[1]-allocations[0])+(1-allocations[1].sum())*((p-allocations[1])/(1-allocations[1].sum()))
        require(tv(reconstructed,p)<1e-15,"telescoping_mass")
        rows=[]
        for ds,values in (("FineWeb",[1.,3.]),("RULER",[10.])):
            for i,value in enumerate(values):
                rows.append({"dataset":ds,"task":"test","context":128,"prompt_id":str(i),"metric":value})
        require(abs(balanced_mean(rows,"metric")-6)<1e-15,"equal_dataset_weights")
        q=weighted_quantile([1,2,3],[1,1,8],[.5,.9,.99])
        require(q==[3.,3.,3.],"weighted_quantiles")
        seed_rows=[{"model":"fixture","policy":"test","method":"test","dataset":"FineWeb",
                    "task":"continuation","context":128,"prompt_id":"a","prefix_index":0,
                    "seed":seed,"conditional_tv":value} for seed,value in ((17,0.0),(31,0.02))]
        prefix_means=average_prefix_seeds(seed_rows,"conditional_tv")
        require(len(prefix_means)==1 and abs(prefix_means[0]["conditional_tv"]-.01)<1e-15,
                "seed_average_before_prefix_quantile")
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"counter.csv"
            path.write_text('==PROF== Connected\n"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                '"0","1","x","dram__bytes_read.sum","byte","1,024"\n'
                '"0","1","x","dram__bytes_write.sum","byte","512"\n')
            require(parse_ncu_csv(path)["device_bytes"]==1536,"hardware_counter_parser")
            output=Path(temp)/"table.tex"
            write_table(output,"lr",["Name","Metric"],[["Example","1.0"]],"Test.","tab:test")
            require("\\begin{table*}" in output.read_text(),"latex_export")
        result={"status":"passed","checks":dict(checks),"total_assertions":sum(checks.values()),
                "torch_version":torch.__version__,"device":"cpu","model_weights_loaded":False,
                "full_checkpoint_cuda_test":"not_run","hardware_profile_test":"CSV parser only",
                "transformers_integration_test":"Run --integration-test in the installed environment"}
        print(json.dumps(result,indent=2))
        return result
    finally:torch.set_num_threads(previous_threads)


def integration_test() -> dict:
    """Download-free native HF integration tests using tiny randomly initialized models."""
    from transformers import LlamaConfig,Qwen3Config,AutoModelForCausalLM
    class Tokenizer:
        pad_token_id=0;eos_token_id=2
        def __call__(self,text,return_tensors="pt"):
            return {"input_ids":torch.tensor([[1]+[(ord(c)%58)+3 for c in text[:15]]])}
    cfg=Config(device="cpu",dtype="float32",interval_chunk=16,leaf=4,initial=8,stages=(1,2),
               warmups=0,repeats=1,contexts=(32,),prefixes=2)
    report={}
    for cls in (LlamaConfig,Qwen3Config):
        conf=cls(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
                 bos_token_id=1,eos_token_id=2,pad_token_id=0)
        model=AutoModelForCausalLM.from_config(conf,attn_implementation="sdpa").eval()
        tok=Tokenizer();runner=validate_native_adapter(model,tok,cfg)
        ids=[1,6,8,11,4,9,12,7,3,6]
        for method in (Method("dense"),Method("quest",fraction=.5),Method("magicpig",bits=3,tables=8),
                       Method("progressive",degree=0,stages=2),Method("progressive",degree=1,stages=2)):
            runner.prepare(ids,method,13,5)
            for j in range(2):
                out=runner.step(ids[-1] if j==0 else 5,method,Policy("topk",5,.7),np.random.default_rng(13))
                with torch.inference_mode():
                    dense=runner.dense_logits(runner.last_state)
                    runner.audit(runner.last_state,method,Policy("topk",5,.7),dense)
                assert 0<=out.token<64
                assert int(runner.state.get_seq_length())==len(ids)+j
        runner.close();report[conf.model_type]="passed"
    result={"status":"passed","native_hf_tiny_models":report,"pretrained_model_measurements":False}
    print(json.dumps(result,indent=2));return result


def parse_args() -> tuple[Config,argparse.Namespace]:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models",nargs="+",help="qwen, llama, both, or exact model identifiers; default MODEL_NAME")
    parser.add_argument("--revision",default=MODEL_REVISION)
    parser.add_argument("--suite",choices=("smoke","paper"),default="smoke")
    parser.add_argument("--datasets",nargs="+",default=["fineweb"],help="fineweb longbench ruler, or all")
    parser.add_argument("--out",default=OUTPUT_DIR)
    parser.add_argument("--device",default=DEVICE)
    parser.add_argument("--dtype",choices=("bfloat16","float32"),default=MODEL_DTYPE)
    parser.add_argument("--contexts",type=int,nargs="+")
    parser.add_argument("--n-eval",type=int)
    parser.add_argument("--n-cal",type=int)
    parser.add_argument("--max-long-examples",type=int)
    parser.add_argument("--prefixes",type=int)
    parser.add_argument("--warmups",type=int)
    parser.add_argument("--repeats",type=int)
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--temperature",type=float,default=TEMPERATURE)
    parser.add_argument("--top-k",type=int,default=TOP_K)
    parser.add_argument("--top-p",type=float,default=TOP_P)
    parser.add_argument("--leaf-size",type=int,default=LEAF_SIZE)
    parser.add_argument("--initial-block-size",type=int,default=INITIAL_BLOCK_SIZE)
    parser.add_argument("--interval-chunk",type=int,default=512)
    parser.add_argument("--bootstrap",type=int)
    parser.add_argument("--ruler-dir",default=RULER_DIRECTORY)
    parser.add_argument("--prepare-ruler",action="store_true",help="Download and run the official NVIDIA data generator")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--no-task-generation",action="store_true",help="Skip own-continuation task scores and teacher-forced NLL")
    parser.add_argument("--no-eval-grid",action="store_true",help="Evaluate selected points plus width and degree diagnostics")
    parser.add_argument("--aggregate-only",action="store_true")
    parser.add_argument("--fill-tex",help="Write a measured copy replacing supported table labels, preserving other content")
    parser.add_argument("--profile-hbm",action="store_true",help="Profile already-selected methods with Nsight Compute")
    parser.add_argument("--profile-prompts",type=int,default=1,help="Prompts per dataset/task/length stratum; zero means all")
    parser.add_argument("--profile-prefixes",type=int,default=1)
    parser.add_argument("--ncu",default="ncu")
    parser.add_argument("--counter-worker",help=argparse.SUPPRESS)
    parser.add_argument("--allow-other-transformers",action="store_true",help="Bypass the tested Transformers release range while retaining API checks")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--integration-test",action="store_true")
    args=parser.parse_args()
    names=args.models or [MODEL_NAME]
    if names==["both"]:names=list(ALIASES.values())
    else:names=[ALIASES.get(name,name) for name in names]
    datasets=["fineweb","longbench","ruler"] if "all" in args.datasets else args.datasets
    if any(name not in {"fineweb","longbench","ruler"} for name in datasets):parser.error("Unknown dataset")
    cfg=Config(models=tuple(names),revision=args.revision,suite=args.suite,datasets=tuple(datasets),
               out=args.out,device=args.device,dtype=args.dtype,temperature=args.temperature,
               top_k=args.top_k,top_p=args.top_p,leaf=args.leaf_size,initial=args.initial_block_size,
               interval_chunk=args.interval_chunk,ruler_dir=args.ruler_dir,prepare_ruler=args.prepare_ruler,
               generate_tasks=not args.no_task_generation,eval_grid=not args.no_eval_grid,
               profile_prompts=args.profile_prompts,profile_prefixes=args.profile_prefixes,ncu=args.ncu,
               allow_other_transformers=args.allow_other_transformers)
    if cfg.suite=="smoke":
        cfg.contexts=(4096,);cfg.n_eval=2;cfg.n_cal=1;cfg.max_long_examples=2
        cfg.prefixes=2;cfg.warmups=0;cfg.repeats=1;cfg.bootstrap=200;cfg.seeds=(17,)
        cfg.quest_fractions=(.25,1.0);cfg.magic_settings=((9,128),)
    for name in ("n_eval","n_cal","max_long_examples","prefixes","warmups","repeats","bootstrap"):
        value=getattr(args,name)
        if value is not None:setattr(cfg,name,value)
    if args.contexts is not None:cfg.contexts=tuple(sorted(set(args.contexts)))
    if args.seeds is not None:cfg.seeds=tuple(args.seeds)
    if cfg.temperature<=0 or not 0<cfg.top_p<1 or cfg.top_k<1:parser.error("Invalid decoding settings")
    if cfg.leaf<1 or cfg.initial<cfg.leaf or cfg.initial%cfg.leaf:parser.error("Initial block size must be a multiple of leaf size")
    if cfg.n_eval<1 or cfg.n_cal<1 or cfg.prefixes<1 or cfg.repeats<1 or cfg.warmups<0:parser.error("Invalid sample or repetition count")
    if not cfg.contexts or min(cfg.contexts)<32:parser.error("Context lengths must be at least 32")
    if not cfg.seeds:parser.error("At least one random seed is required")
    return cfg,args


def main() -> int:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",datefmt="%H:%M:%S")
    cfg,args=parse_args()
    if args.self_test:self_test();return 0
    if args.integration_test:integration_test();return 0
    if args.counter_worker:counter_worker(Path(args.counter_worker));return 0
    if args.aggregate_only or args.profile_hbm:
        path=Path(cfg.out)/"config.json"
        if not path.exists():raise FileNotFoundError(f"Missing existing experiment configuration: {path}")
        saved=Config(**json.loads(path.read_text()))
        saved.out=cfg.out;saved.ncu=cfg.ncu
        saved.profile_prompts=cfg.profile_prompts;saved.profile_prefixes=cfg.profile_prefixes
        if args.profile_hbm:profile_hardware(saved)
        if args.aggregate_only or args.fill_tex:aggregate(saved,args.fill_tex)
        return 0
    if args.prepare_only:
        Path(cfg.out).mkdir(parents=True,exist_ok=True)
        config_path=Path(cfg.out)/"config.json"
        if config_path.exists():
            saved=Config(**json.loads(config_path.read_text()))
            if config_identity(saved)!=config_identity(cfg):
                raise RuntimeError("This output directory belongs to a different experiment. Use another --out directory.")
        else:atomic_json(config_path,dataclasses.asdict(cfg))
        prepare_data(cfg);return 0
    run(cfg,args.fill_tex)
    return 0


if __name__=="__main__":
    try:
        code=main()
    except KeyboardInterrupt:
        code=130
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Experiment stopped. Unmeasured or invalid results were not replaced with estimates.")
        code=1
    finally:
        # Run Python-level destructors while the interpreter and GIL are still active.
        gc.collect()
    raise SystemExit(code)
