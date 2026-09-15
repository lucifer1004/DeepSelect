# DeepSelect

DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) (which is used in DeepSeek V3.2, DeepSeek V4, and DeepSeek V4.1 models) and the sampler. The upstream SM100 performance results report 2 ~ 20x speedup compared to vanilla `torch.topk`; these are not SM120/SM121 measurements.

## News

- 2026.09.10: We've released a brief analysis of the algorithm and its implementation: [English](docs/DeepSelect-deep-dive.md) | [中文](docs/DeepSelect-deep-dive.zh.md)
- 2026.09.10: We've released DeepSelect v1.0.0

## Supported Cases

### Native SM120 / SM121

This port adds native CUDA selection for SM120 (`12.0`) and SM121 (`12.1`)
while retaining the default SM100/SM103 kernels. The implementation uses the
same native algorithms for both SM12 targets, with dispatch enabled only for
the exact target compiled into the extension. Other SM12 capabilities are
rejected; there is no `torch.topk` fallback. Explicit SM121 compilation is not
GPU qualification: SM121 hardware validation remains pending. This port does
not claim validated SM12 performance.

The public `deep_select.topk` API supports BF16 and FP32 input, int32/int64
indices, `1 <= topk <= 4096`, `0 <= vocab_size < 2**23`, and empty batches;
`batch_size` must fit int32. Zero-width rows are supported on SM120/SM121;
SM100/SM103 require `vocab_size > 0` for nonempty batches. Empty batches remain
supported on all compiled targets. `sorted=True` sorts by descending value (FP32 only,
requires `return_value=True`), while `sorted_index=True` sorts by ascending
index. These options are mutually exclusive. `begin` and `hint` remain
unsupported. Selection, optional values, sorting, variable row lengths, and
index offsets execute in CUDA.

Unsorted int32 `topk == 512` with `2048 < vocab_size <= 131072` uses a
256-thread streaming selector. It scans 1024 inputs per tile in deterministic
bit-reversed tile order and filters against the exact worst retained key/index
pair. A bounded 2560-candidate buffer is compacted to 512 when its size reaches
1536. Candidate radix selection uses warp-private histograms and original-index
tie breaking; explicit shared storage is bounded by 32 KiB. Tile ordering is
not a worst-case performance guarantee.

Within that K512 path, `65536 <= vocab_size <= 131072` and
`1 <= batch_size <= 16` use eight independent blocks per row followed by one
exact merge block. Local selection and merge retain original indices, without
intermediate sorting or index remapping. Per-call device scratch is
`batch_size * 8 * 1025` int32 words for keys, indices, and valid counts.
Allocation and both launches use the current stream and support CUDA graph
capture. Other streaming shapes use one block per row. Selection scans each
input once, with original-value gathers when values are requested.

Other unsorted int32 configurations with `topk <= 512` use one 256-thread
block per row, whole-row 8-bit radix histograms, parallel pivot selection, and
scan-based collection. BF16 needs at most two value radix passes and FP32 at
most four; exact bucket completion can stop refinement early. Explicit shared
storage is bounded by 4 KiB without sort scratch. Other configurations use a
512-thread fallback with four radix passes and optional CUB sorting of at most
4096 packed keys, reusing shared storage with a compile-time bound of 48 KiB.
Compiler-generated shared storage is additional to these explicit bounds.
None of the native SM12 paths uses TMA or the upstream 16-block cluster.

Equal values prefer the lower original index. `+0` and `-0` compare equal and
retain their original bit patterns in returned values; infinities and
subnormals are ordered numerically. Unsorted output order is unspecified,
except for short rows as described below. Optimized paths retain the NaN
contract below, and bounded output stores write only logical output elements.

TopK workloads vary widely, and the fastest algorithm & implementation highly depends on the input dtype, `batch_size`, `vocab_size`, and `topk`. This repository only focuses on the following cases:

### Lightning Indexer Scenario

This scenario covers:
- Input dtype: `torch.bfloat16`
- `batch_size`: $1 \sim +\infty$ (both large and small batch sizes are optimized)
- `vocab_size`: $1 \sim +\infty$ (both large and small vocabularies are optimized)
- `topk`: small (must be $\le 4096$; larger values are not supported)

Recommendations:
- Disable `sorted_index` unless the output has to be ordered by index or by value; enabling either one costs performance.
- Set `return_value=False` when the values are not needed. This skips the value output and is faster.

### Sampling Scenario

This scenario covers:
- Input dtype: `torch.float32`
- `batch_size`: $1 \sim +\infty$
- `vocab_size`: around 128K
- `topk`: small (must be $\le 4096$; larger values are not supported)

## Performance

The figures below are inherited upstream SM100 results, not SM120/SM121
benchmarks. No SM12 speedup is established by this port.

Measured with the benchmark in [`tests/test.py`](tests/test.py)
(`python3 tests/test.py --perf-only`), which reports the ratio against `torch.topk`
on the same input. The metric is effective memory bandwidth: TopK does no
floating-point math, so a FLOP rate would not be meaningful here.

### Lightning Indexer Scenario

bfloat16, `topk = 512`, one subplot per batch size, on a shared 0 - 7 TB/s axis.

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer](assets/perf_bf16.png)

### Sampling Scenario

float32, `vocab_size = 129280`, `topk = 512`.

![DeepSelect vs torch.topk, float32 Sampling](assets/perf_fp32.png)

## Installation

```bash
git clone --branch dev https://github.com/vllm-project/DeepSelect.git
cd DeepSelect
git submodule update --init --recursive
pip install -v .
```

### CUDA architecture selection

The default source build targets `10.0a;10.3a`. Set
`DEEP_SELECT_CUDA_ARCH_LIST` to select explicit targets from `10.0a`, `10.3a`,
`10.0f`, `12.0`, and `12.1`, for example `12.0;12.1` or
`10.0a;10.3a;12.0;12.1`. Entries may be separated by spaces or semicolons.
`+PTX` adds matching PTX alongside the native cubin, for example `12.1+PTX`.
Unsupported entries are errors; `TORCH_CUDA_ARCH_LIST` does not replace this
package's default.

SM120 (`12.0`) and `10.0a` require CUDA 12.8 or newer; `10.3a`, `10.0f`, and
SM121 (`12.1`) require CUDA 12.9 or newer. The [CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html#general-cuda)
explicitly introduce compiler target support for `sm_121`; compiler support
alone does not establish runtime qualification.

SM120 and SM121 use the same native kernel source under `csrc/cuda_kernels/sm120`,
with the same dtype, top-k, sorting, alignment and unsupported-option limits
in `deep_select/interface.py`. Explicit SM121 selection emits `sm_121`, rather
than relying on an SM120 cubin's forward compatibility. The SM100/103 v3 and
cluster kernels never compile for either SM12 target. The SM12 family disables
fast math and preserves subnormal values (`--ftz=false`).

The host API compiles once and all selected objects link into one shared library,
`deep_select.deep_select_cuda`, retaining `PyInit_deep_select_cuda`. The build
targets the PyTorch 2.10 stable ABI and CPython 3.10+ (`cp310-abi3`). Importing
this extension registers `torch.ops.deep_select.topk` and
`torch.ops.deep_select.get_alignment_requirement`; the Python API uses those
operators. This is one provider with one operator registration, not separate
SM100 and SM12 Python extension modules.

The build defines `DEEP_SELECT_BUILD_SM100`, `DEEP_SELECT_BUILD_SM120`, and
`DEEP_SELECT_BUILD_SM121` according to the selected targets. Legacy source
consumers retain defaults of `1`, `0`, and `0`, respectively. Builds must pass
the artifact spill checker: LOCAL must be zero and STACK must not exceed 8
bytes. Compiler target support alone is not runtime or performance qualification.

The existing vLLM source consumer still targets `10.0f` and uses its existing
source glob. This standalone fork port does not enable SM12 in vLLM: separate
consumer integration is needed to include and compile the native SM12 sources
and select the matching targets while retaining a single operator provider.

## Usage

```python
import torch
import deep_select

# input: (batch_size, vocab_size), torch.bfloat16 or torch.float32.
# Its row stride must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and its last dimension must be contiguous.
batch_size, vocab_size, topk = 4, 204800, 1024

x = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device="cuda")

values, indices = deep_select.topk(
    x,
    topk,
    sorted_index=True,         # return each row's indices in ascending order
    indices_type=torch.int32,  # torch.int32 or torch.int64
    return_value=True,         # False skips the value output
)
# values:  (batch_size, topk) of x.dtype
# indices: (batch_size, topk) of indices_type
```

All tensor arguments must be on the same CUDA device as `input`. Execution
uses that device's current PyTorch CUDA stream, including non-default streams
and CUDA graph capture, and restores the caller's current device afterward.

The input's last dimension must be contiguous, its row stride must be aligned
to `deep_select.get_stride_requirement()[0]` bytes, and its data pointer must be
32-byte aligned. Nonempty backing storage must include the final row rounded
up to 128 bytes for the upstream TMA kernels; SM12 retains the same validation.
Allocate padded rows first, then slice to the logical width: stride alignment
alone does not ensure final-row padding.

Outputs have strides aligned to `deep_select.get_stride_requirement()[1]`
bytes and may be non-contiguous. `return_value=False` returns `(None, indices)`.
Pass `output_idx=` to supply a `(batch_size, topk)` buffer of `indices_type`;
it must have a contiguous last dimension, aligned row stride and 32-byte-aligned
pointer, and non-overlapping rows. Only logical output elements are written,
so trailing allocation padding is not required for outputs.

Nonempty outputs must not share or overlap backing storage with `input`, `end`,
`output_idx_offset`, or each other. Even disjoint views of the same backing
storage are rejected, regardless of dtype or storage offset; allocate output
buffers separately. Pairs containing an empty tensor are exempt. These checks
also apply to direct `torch.ops.deep_select.topk` calls, including a supplied
`output_value` when `return_value=False`.

For the full signature, see [`deep_select/interface.py`](deep_select/interface.py).

### Variable-length rows

`end` is a contiguous `(batch_size,)` int32 CUDA tensor of per-row exclusive
upper bounds. The caller must ensure `0 <= end[i] <= vocab_size`; host
validation does not read CUDA tensor contents. `output_idx_offset` is an
optional `(batch_size,)` int32 CUDA tensor added only to valid indices, never
to fill entries.

When `end[i] <= topk`, valid entries precede all fill entries and only the valid
prefix is sorted, even if the fill value is larger than the input values.
Padding uses `value_oob_fill_value` (default `-inf`) and `idx_oob_fill_value`
(default `2147483647`). The index fill must fit int32; the value fill may be a
float32-representable finite value, infinity, or NaN.

```python
batch_size, vocab_size = 2, 129280   # 129280 is a multiple of 256, so float32 is fine
x = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda")

end = torch.tensor([129280, 100000], dtype=torch.int32, device="cuda")  # (batch_size,)
values, indices = deep_select.topk(x, 1000, end=end, sorted=True,
                                   indices_type=torch.int64)
```

### NaN handling

NaN checking remains enabled in optimized paths. With the default
`abort_when_nan_found=True`, the kernel invokes `trap()` and aborts. With
`False`, index zero of the affected output row is set to `0x3F3F3F3F` without
adding the index offset; the rest of that row's outputs are unspecified.
Other rows continue normally.

Rows whose length is `<= topk` are never NaN-checked. On SM12, sorting such a
short row by value puts NaNs first (ties by original index), preserves their
input payloads in returned values, and keeps fill entries at the end.
Unsorted short rows preserve input order.

The input-device/stream guards and vector-boundary safety changes were adapted
from upstream [PR #4](https://github.com/deepseek-ai/DeepSelect/pull/4)
(`a24b1e4`) and [PR #6](https://github.com/deepseek-ai/DeepSelect/pull/6)
(`c799e83`), both authored by [morluto](https://github.com/morluto).

## Citation

```text
@misc{deepselect2026,
    title={DeepSelect: High-Performance TopK Kernels for DeepSeek Sparse Attention and Sampling},
    author={Yi Qian and Shengyu Liu and Yichen Li},
    year={2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/deepseek-ai/DeepSelect}},
}
```
