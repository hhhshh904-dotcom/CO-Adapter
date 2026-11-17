# CO-Adapter: Reading Each Modality First

A small, single-device research implementation of four modality-local input
tokens for a frozen ChatGLM3-6B-base. **Content-only (`co`) is the default**;
`mixed` is an optional controlled variant. There is no research-worker import
chain, GPU scheduler, prediction cache, bundled model, or bundled dataset.

**Validation status:** CPU unit/integration tests and tiny-model contracts against
the external official implementation have been run. This reorganized package has
**not** been validated by end-to-end training or smoke testing with the real 6B
model on a GPU. It is not a claim of bitwise reproduction of historical runs or
of the reference paper's environment/results. SIMS-V2 was used during method
development; its test split is not an untouched confirmatory evaluation.

## Layout

```text
train.py                    # CLI usable without installation
src/co_adapter/
  adapter.py                # temporal memory, independent attention, four tokens
  upstream.py               # external official imports and runtime path overrides
  engine.py                 # training, resume, exact fresh-reload/test gate
  state.py                  # atomic checkpoints and RNG state
  provenance.py             # current source and dependency inventory
  cli.py                    # smoke / train / evaluate
  __init__.py, __main__.py
tests/                      # unittest fixtures and optional official CPU contracts
requirements.txt, pyproject.toml
LICENSE, THIRD_PARTY_NOTICES.md, docs/REPRODUCIBILITY.md
```

## Dependencies and external assets

Use **Python 3.11**. `requirements.txt` records the core versions present in the
CPU-tested environment (PyTorch was `2.1.2+cu121`). These are not a full environment
lock. Fresh dependency resolution/installation was **not** tested; choose the
appropriate PyTorch wheel for your platform and check the resulting environment.
The installed pandas/scikit-learn metadata requires Python >=3.11 and permits
NumPy 1.26.4; this package deliberately limits its declared Python support to 3.11.

For an already prepared environment, no installation is needed:

```bash
python train.py --help
python -m unittest discover -s tests -v
```

Optional editable installation in an environment with dependencies already
installed (this does not fetch dependencies or build tools):

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m co_adapter --help
```

Obtain these assets **separately**, respecting their licenses/access conditions:

1. The official [MSE-Adapter repository](https://github.com/AZYoung233/MSE-Adapter).
   Pass its `MSE-ChatGLM3-6B` directory as `--upstream-root`. It must contain
   `models/multiTask/CMCM.py`, `models/subNets/Textmodel.py`, the customized
   `models/ChatGLM3`, `data/load_data.py`, `config`, and `utils`.
   A generic Hugging Face ChatGLM implementation is not a replacement for the
   customized `input_fusion` model. Official source is imported externally,
   not copied into this release; see [third-party notices](THIRD_PARTY_NOTICES.md).
2. **ChatGLM3-6B-base** weights/tokenizer in a local directory passed through
   `--model-path`, not the conversational checkpoint. The official loader
   executes trusted model/tokenizer code; inspect external assets before use.
3. The official processed data, with this layout under `--data-root`:

   ```text
   SIMS_V2/ch-simsv2s.pkl
   MOSEI/Processed/unaligned_50.pkl
   MELD/meld_train.pkl, meld_valid.pkl, meld_test.pkl
   CHERMA0723/cherma_train.pkl, cherma_valid.pkl, cherma_test.pkl
   ```

The official README links the processed datasets and model resources. This CLI
operates on local paths and sets Hugging Face offline mode; it does not download
assets. It overrides `dataPath`, dataset root and model path in memory, including
upstream revisions with hardcoded resource paths. It never edits external files.
Only use trusted pickle datasets and checkpoints: deserialization can execute code.

## Protocol retained

- Construct official CMCM first, including its temporary mixer/fusion allocation;
  replace only that adapter path. Preserve post-construction RNG restoration.
- Reuse each official audio/video LSTM, dropout and 256-dimensional projection,
  exposing every declared valid temporal state, not only the final state.
- Two learned 128-wide queries per modality, four attention heads and separate
  output projections: audio query 1/2 followed by video query 1/2, then transcript
  embeddings. CO does not condition either query on text; mixed conditions only
  the second query per modality on the masked transcript mean. Text still enters
  the LLM in both variants; no semantic disentanglement is claimed.
- Keep the **unused text projection** in CO: allocation, initialization RNG draws,
  and checkpoint names are preserved. For SIMS-V2, trainable allocated state is
  **1,866,624** parameters; active CO state is **1,342,208**. The retained
  projection accounts for 524,416 parameters and receives no CO gradient.
- Frozen FP16 backbone; FP16 autocast and GradScaler on CUDA, batch 8, one update
  per batch, AdamW with constant LR `5e-5`, epsilon `1e-4`, betas `(0.9, 0.999)`,
  weight decay `.01`; unscale before gradient norm clipping at `1.0`. No scheduler
  or gradient accumulation. CPU tests use FP32 tiny fixtures, not a 6B CPU claim.
- Import official prompts, tokenization, wrapping, supervision, parser and metrics
  unchanged. **No added answer EOS, no masking out answer-padding targets, no
  new LLM transcript-padding attention mask, no parser normalization rewrite.**
  Regression targets use one decimal place; classification uses class IDs.
  Invalid parsed answers fall back to zero upstream. Save raw decoded text and
  parse-failure counts; reject nonfinite returned predictions rather than hide them.
- Validation and test use greedy official stream generation with only
  `use_cache=False` overridden, recomputing the full prefix at every step. Keep
  upstream EOS/stopping behavior; no incremental KV patch. All splits use
  shuffled loaders, including validation and test.

| Dataset flag | Selection | Patience | Maximum new tokens |
| --- | --- | --- | --- |
| `simsv2` | minimum validation MAE | 10 | 4 |
| `mosei` | minimum validation MAE | 10 | 4 |
| `meld` | maximum validation weighted F1 (`weight_F1`) | 8 | 2 |
| `cherma` | maximum validation weighted F1 (`weight_F1`) | 8 | 2 |

The first finite validation result is eligible; subsequent improvements must be
at least `1e-6`. Stop after the dataset patience without improvement; there is no
arbitrary epoch cap. Dataset-specific official metric clipping/rounding remains
in force. No validation NLL selection or auxiliary loss is introduced.

## Run sequence

From the release root, set local directories and expose **one** GPU. No multi-GPU
scheduling is implemented. The commands below describe the intended real-model
acceptance workflow; **they have not been GPU-tested in this release**.

```bash
export CUDA_VISIBLE_DEVICES=0
UPSTREAM=/path/to/MSE-Adapter/MSE-ChatGLM3-6B
MODEL=/path/to/chatglm3-6b-base
DATA=/path/to/processed-data
OUT=outputs/simsv2/co/1111

COMMON=(--upstream-root "$UPSTREAM" --model-path "$MODEL" --data-root "$DATA"
        --output-dir "$OUT" --dataset simsv2 --seed 1111 --device cuda:0)

# 64 training examples, 8 validation examples, one epoch; never loads test.
python train.py smoke "${COMMON[@]}"
# A separate invocation builds a fresh model and exactly replays smoke validation.
python train.py evaluate "${COMMON[@]}" --smoke-only
# Training requires the successful smoke + separate reload above.
python train.py train "${COMMON[@]}"
# Explicit held-out access: replay full best-validation JSON first, then test once.
python train.py evaluate "${COMMON[@]}"
```

Use `--variant mixed` on **every** command and a different output directory for
mixed queries. Other dataset flags use the same sequence. Any integer seed is
accepted; the original controlled runs used 1111, 2222 and 3333. Each output
configuration needs its own smoke/reload gate. Installed usage is
`python -m co_adapter ...` or `co-adapter ...` with the same arguments.

Resume an interrupted training or smoke invocation with its original arguments:

```bash
python train.py train "${COMMON[@]}" --resume
# Or: python train.py smoke "${COMMON[@]}" --resume
```

SIGINT/SIGTERM request saving after the current completed training batch.
`latest.pt` includes trainable model parameters, model buffers, AdamW state,
GradScaler state, Python/NumPy/Torch/CUDA RNG, epoch/phase, completed-batch offset
and epoch-start loader RNG. Reconstruct the shuffled loader and skip completed
batches before restoring current RNG. Dataset retrieval is assumed deterministic,
with zero loader workers as in the official pipeline. Unexpected hard termination
can replay work since the most recent checkpoint (every 25 batches and at phase
boundaries); graceful interruption saves the current completed batch. Frozen
backbone weights remain external, not duplicated in every checkpoint.

`best.pt` stores the selected parameters/buffers and validation-start RNG.
Evaluation requires completed training and a **fresh process**, recreates the
model from external frozen weights, restores best state and validation RNG,
and compares the **entire** prediction/metric JSON with its saved best-validation
record. Only then can explicit `evaluate` load test. `--smoke-only` never accesses
test. No previous replay marker bypasses the new replay, and existing test output
cannot be overwritten. No GPU test results are shipped.

`run.json` records settings, resource paths, hashes of current package and external
upstream Python sources, and core dependency versions. Changed recorded source,
resources or dependencies reject resume/reload; no historical hashes/manifests or
execution-plan chain is required. Model/data **contents are not hashed**: keep
those directories immutable yourself. Paths must remain unchanged within a run.
This inventory is not a cryptographic authenticity or full environment guarantee.

Use one writer per output directory. A strict exact-replay failure on a different
GPU/software stack is a real gate failure, not permission to loosen comparison or
reuse predictions. Real-model GPU memory/throughput requirements and portability
remain unmeasured for this reorganized package.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v

# Optional: imports actual official code, but no pretrained weights or real data.
CUDA_VISIBLE_DEVICES='' CO_UPSTREAM_ROOT="$UPSTREAM" \
  python -m unittest discover -s tests -v
```

Tests cover CO/mixed paths, padding/length masks, frozen-gradient flow, retained
allocation, constant optimizer settings, exact interrupted/resumed state,
selection/patience, same-process rejection, fresh-process smoke/train/reload,
corrupted-validation test denial, and no-cache calls. Optional official contracts
check all four runtime configs, original no-EOS/padded answer supervision/parser,
and a randomly initialized tiny actual ChatGLM's shifted CE and increasing
full-prefix lengths with no cache. These are software-contract tests, **not**
real-data accuracy checks or evidence of completed 6B GPU experiments.

See [reproducibility and review limits](docs/REPRODUCIBILITY.md) before interpreting
results or preparing an anonymous venue submission.
