# vLLM-infereence-optimization-on-T4-GPU
# vLLM Inference Optimization on T4 GPUs

Step-by-step serving optimizations for Qwen2.5-3B-Instruct on NVIDIA T4s (Kaggle), measuring tokens/sec at each step.

## 1. Quantization: FP16 vs AWQ 4-bit (5 prompts)

| Model | tok/s |
|---|---|
| FP16 | 138.5 |
| AWQ 4-bit | 328 (**2.4x**) |

AWQ still answered correctly (17 × 23 = 391). Generation is memory-bound, so 4x smaller weights mean far less to read per step.

## 2. Batching: max_num_seqs (AWQ, 64 prompts, 1 GPU)

| max_num_seqs | tok/s |
|---|---|
| 4 | 310 |
| 16 | 911 |
| 64 | 1,361 (**4.4x vs 4**) |

Bigger batches reuse each weight read for more answers. The gains shrink as the GPU shifts from memory-bound to compute-bound.

## 3. Data parallel: one model copy per GPU (2x T4)

| Total prompts | GPU 0 | GPU 1 | Total tok/s |
|---|---|---|---|
| 64 (32 per GPU) | 937 | 975 | 1,911 (1.4x) |
| 128 (64 per GPU) | 1,657 | 1,670 | **3,327 (2.4x)** |

Verified with `nvidia-smi`: both GPUs at 100% utilization, about 11.6 GB each.

## Key takeaways
- Quantize first: AWQ 4-bit gave 2.4x with no visible quality loss.
- Batch aggressively: the same GPU did 4.4x more work with larger batches.
- Extra GPUs only help with enough traffic: half-full batches gave 1.4x; full batches gave 2.4x over one GPU.

## 5. Cost per million tokens

Assumes a cloud T4 at about $0.35/hour per GPU (prices vary by provider).

| Setup | tok/s | Tokens per hour | Cost per 1M tokens |
|---|---|---|---|
| FP16, 5 prompts | 138.5 | 0.5M | ~$0.70 |
| AWQ 4-bit, 5 prompts | 328 | 1.2M | ~$0.30 |
| AWQ, batch 64 | 1,361 | 4.9M | ~$0.07 |
| AWQ, 2x T4, 128 prompts | 3,327 | 12M | ~$0.06 |

**Takeaway:** quantization plus batching cut serving cost about 10x on the same hardware. Cost per token depends more on how well you use the GPU than on the GPU itself.
