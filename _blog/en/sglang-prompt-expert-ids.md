---
layout: blog-post
lang: en
translation_url: /blog/zh/sglang-prompt-expert-ids/
date: 2026-11-27
title: "Where do the prompt’s expert IDs go in an SGLang MoE rollout?"
excerpt: Two rollouts produced the same tokens and returned expert tensors of the same shape. Why did the first eight rows turn to zeros when we ran prefill and decode on separate workers?
---

In reinforcement learning with a mixture-of-experts (MoE) model, the rollout and training forward passes can route the same token to different experts. Rollout Routing Replay (R3) reduces this mismatch by recording the experts selected for each token at each layer during rollout and replaying those choices during training. To use R3 with SGLang, we therefore need the inference engine to return both the generated tokens and the expert IDs it actually selected. [R3 paper](https://arxiv.org/abs/2510.11370)

SGLang exposes these IDs through its native `/generate` endpoint. Setting `return_routed_experts=true` makes the trace available in `meta_info.routed_experts`. With prefill–decode (PD) disaggregation, two workers hold different parts of the trace. The prefill worker processes the prompt, and the decode worker runs the subsequent generation steps. Which positions the trainer can replay depends on how the HTTP router assembles the response.

We tested this with Qwen3-30B-A3B-Base. We fixed the weights, used greedy decoding, and asked the model to generate four tokens from an eight-token prompt. We saved the expert IDs from a single-worker deployment as our reference, then sent the same non-streaming request to a disaggregated deployment. Both runs generated `[279, 1909, 315, 847]`. Both returned an expert tensor of shape `[11, 48, 8]`. The discrepancy was already visible in the first layer’s top eight experts for the first prompt token.

```text
Single worker    [11, 0, 64, 79, 118, 86, 122, 38]
Disaggregated    [ 0, 0,  0,  0,   0,  0,   0,  0]
```

Comparing the tensors one token at a time revealed a sharp boundary. In the disaggregated response, all eight prompt rows contained only zeros. The final three rows, corresponding to generated tokens, matched the reference entry for entry. The boundary fell exactly where the prefill worker handed the first generated token to the decode worker.

## Eight prompt tokens and three decode steps

The prefill worker processes all eight input tokens in one forward pass. At the last position it predicts token 279, then passes that token and the attention key–value (KV) cache to the decode worker. The decode worker feeds 279 back into the model to predict 1909, feeds in 1909 to predict 315, and feeds in 315 to predict 847.

Counting the inputs to each forward pass tells us which worker owns each routing decision. Prefill processed eight prompt tokens. Decode processed 279, 1909, and 315. Generation stopped as soon as the model produced 847, so no forward pass ever took 847 as input. Twelve tokens appeared in the request and its completion, but only eleven had routing records.

| Source | Prompt P1–P8 (IDs 1000–1007) | G1 (279) | G2 (1909) | G3 (315) | G4 (847) |
|:---|:---|:---|:---|:---|:---|
| Single worker | R1–R8 | R9 | R10 | R11 | — |
| Prefill worker | R1–R8 | — | — | — | — |
| Decode worker | All zeros | R9 | R10 | R11 | — |
| HTTP response before the fix | All zeros | R9 | R10 | R11 | — |
| HTTP response with #53355 | R1–R8 | R9 | R10 | R11 | — |

Each R denotes one token’s complete reference record (48 layers × 8 experts). “All zeros” means the worker returned 384 zeros per prompt position. A dash means there was no record.

We inspected `routed_experts` from each worker separately. The prefill worker’s eight rows matched the first eight reference rows exactly. The decode worker ran only three steps, yet returned eleven rows. The eight zero-filled rows were already present in its response.

Those eleven rows came from a read of the decode worker’s local routing buffer at the end of the request. Each worker has its own buffer, initialized to zero. A forward pass writes only the positions that worker processes. Transferring the prompt’s KV cache lets decode continue generation, but the routing decisions for the eight prompt positions still live in the prefill worker’s buffer. [Buffer initialization and writes](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/state_capturer/base.py#L63-L118)

In SGLang v0.5.23, the version we tested, `get_topk` reads a range based on the length of the entire request. Here the range starts at zero and spans `[0, 8 + 4 - 1)`, covering P1–P8 and G1–G3. Decode had never written this prompt’s expert IDs at the first eight positions. In our run, it read zeros there. The last three positions held the routes that decode had computed for G1–G3. [Read range in get_topk](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/state_capturer/base.py#L173-L188)

## Merging expert IDs by token position in the HTTP router

Prefill had already returned the correct expert IDs for the prompt. At the `sgl-router` revision we tested (`8703754`), however, `Http::respond` returned the decode response by default because decode holds the full generated sequence. The router already retrieved input-token log probabilities from prefill when requested. Expert IDs had no equivalent handling. [HTTP response handling](https://github.com/sgl-project/sglang/blob/870375452ae12db9c3db0b1de5a38be2327a54f7/sgl-router/src/server/routes/chat/forward.rs#L516-L529)

The first change in #53355 concerns when the router waits for prefill. For a non-streaming `/generate` request with `return_routed_experts=true`, the router records the token position at which the caller wants the trace to start and waits for the prefill response body. The generated sequence still comes from decode. With both response bodies available, the router can reconstruct `meta_info.routed_experts` separately. [Parsing the request](https://github.com/sgl-project/sglang/blob/494a59616d3acf44db9abafb98aa9df1ce244481/sgl-router/src/server/routes/chat/preparation.rs#L394-L410) · [Waiting for prefill](https://github.com/sgl-project/sglang/blob/494a59616d3acf44db9abafb98aa9df1ce244481/sgl-router/src/server/routes/chat/forward.rs#L344-L353)

How should the router join the traces? Prepending the eight prefill rows to all eleven decode rows would represent the prompt twice. Always skipping the first eight decode rows would make the opposite mistake if the decode worker returned only the three rows it actually computed. In that layout we would discard every valid decode row.

Both layouts share one property. Of the four generated tokens, prefill predicted the first, leaving three forward passes for decode. Whether decode returns eleven rows or three, the rows we need are its **last three**.

```python
prompt_routes = prefill_routes                 # P1 … P8
generated_routes = decode_routes[-3:]          # G1, G2, G3
routes = concatenate([prompt_routes, generated_routes])
```

On the wire, the trace is a base64-encoded byte string. The patch decodes the prefill payload and divides its size in bytes by the number of requested prompt rows (`prompt_tokens - routed_experts_start_len`). This gives the row width without requiring the router to know the layer count or the number of experts selected per layer. The router then uses `completion_tokens - 1` to determine the length of the decode suffix, checks that the decode payload fits one of the two supported layouts, and takes the suffix. Finally, it joins the two byte strings and re-encodes the result. In a batch containing an eight-token prompt and a 23-token prompt, each request gets its own boundary. [Implementation of merge_experts](https://github.com/sgl-project/sglang/blob/494a59616d3acf44db9abafb98aa9df1ce244481/sgl-router/src/server/routes/chat/pd_metadata.rs#L33-L99)

If the model generates only one token, prefill has done all the work. Its trace is sufficient even if the decode response omits the expert field. A generation budget of zero has a different boundary. To match the single-worker read range, the router drops the final prompt position, returning seven rows for our eight-token input.

## Checking the boundary against a single worker

We held the weights, inputs, and sampling procedure fixed, aligned the patched PD response with the single-worker response token by token, and compared the ordered expert IDs at every layer. In the eight-token example, all eight prompt rows and all three decode rows matched exactly. Across mixed-length batches, different trace start positions, and generation budgets of zero or one, all 177 rows in the 15 responses matched the reference. [Reproduction and regression tests](https://github.com/sgl-project/sglang/pull/53355)
