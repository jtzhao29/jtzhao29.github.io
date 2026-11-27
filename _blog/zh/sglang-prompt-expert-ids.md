---
layout: blog-post
lang: zh-Hans
translation_url: /blog/en/sglang-prompt-expert-ids/
date: 2026-11-27
excerpt: 同一个 prompt，两次 rollout 生成了相同的 token，返回的专家张量也形状相同。为什么拆开 prefill 和 decode 之后，前八行专家编号全变成了零？
title: 用 SGLang 做 MoE rollout 时，prompt 的 expert ID 去哪了？
---

> 这个 blog 想要记录 SGLang PR：[sgl-project/sglang#53355](https://github.com/sgl-project/sglang/pull/53355)

SGLang 的 `/generate` 接口可以随生成结果返回每个 token 的 expert ID。我们在官方 HTTP router 上遇到一个问题：开启 prefill/decode（PD）分离后，通过一个非流式请求发送一条 prompt，生成的 token 和非 PD 完全一致，expert 张量的形状也相同，但 prompt 对应的 expert ID 全是零。

这份记录在 prefill worker 上其实是完整的。我们测试的 `sgl-router`（提交 `8703754`）默认返回 decode 的响应，漏掉了 prefill 算出的 prompt 路由。#53355 的修法是，当请求带有 `return_routed_experts=true` 时，router 等待两个 worker 的响应，从 prefill 取调用方要求的 prompt 行，从 decode 取生成部分的行，再写回 `meta_info.routed_experts`。[修复前的返回逻辑](https://github.com/sgl-project/sglang/blob/870375452ae12db9c3db0b1de5a38be2327a54f7/sgl-router/src/server/routes/chat/forward.rs#L516-L529) · [SGLang PR #53355](https://github.com/sgl-project/sglang/pull/53355)

此前 #44002 已经让 prefill worker 返回 prompt 的 expert ID，并用测试客户端验证过怎样合并两段路由。我们的改动把这一步接进官方 HTTP router，rollout 客户端发一次请求就能拿到完整记录。[prefill 侧的工作](https://github.com/sgl-project/sglang/pull/44002)

## 为什么 MoE RL 需要完整的 expert ID

在 MoE RL 里，即使 rollout 和 training 使用同一组 token，两次 forward 也可能选中不同的 expert。R3（Rollout Routing Replay）的做法是在 rollout 时记录每个 token 在各层选中的 expert ID，再在 training 时重放这组选择，以减小 routing 上的 training-inference mismatch。因此，用 SGLang 做 rollout 时，除了生成的 token，我们还需要推理引擎返回当时实际选中的 expert ID。[R3 论文](https://arxiv.org/abs/2510.11370)

这意味着，即使生成文本完全相同，错误的 prompt 路由仍会被训练端当作本次 rollout 的 expert 选择来重放。

## 从一条 8-token prompt 定位缺失的记录

我们用 Qwen3-30B-A3B-Base 做了一个对照。给定一个 8-token prompt，固定权重和 greedy sampling，让模型生成 4 个 token。先在非 PD 部署下保存完整的 expert ID，再把同一个非流式请求发给 PD 部署。两次生成的 token 都是 `[279, 1909, 315, 847]`，返回的 expert 张量形状也都是 `[11, 48, 8]`。但取出第一个 prompt token 在第一层的 top-8，结果已经不同。

```text
非 PD    [11, 0, 64, 79, 118, 86, 122, 38]
PD       [ 0, 0,  0,  0,   0,  0,   0,  0]
```

把两个张量按 token 位置逐行比较，前 8 行对应 prompt，在 PD 响应中全部为零。后 3 行对应生成序列，expert ID 逐项相同。这个分界恰好是 prefill 把第一个生成 token 交给 decode 的位置。

prefill 处理最后一个输入 P8 时预测出 279，随后把 KV cache 和 279 交给 decode。decode 把 279 当作输入，预测出 1909，再用 1909 得到 315，最后用 315 得到 847。

现在可以数清楚每一份专家记录属于谁了。prefill 算过 8 个输入。decode 算过 279、1909、315。至于 847，我们拿到它就结束了请求，模型没有再对它做一次 forward。因此，虽然输入加输出共有 12 个 token，实际应该收到 11 行记录。

![8 个输入和 4 个输出的逐 token 对照。修复前 HTTP 响应和 decode 相同，前 8 行全零。修复后每个位置都与单 worker 对照一致。]({{ "/images/blog/sglang-prompt-expert-ids/pd-experts-token-alignment.png" | relative_url }})

分别取出两个 worker 返回的 `routed_experts`，prefill 的 8 行与非 PD 的前 8 行逐项一致。decode 虽然只执行了 3 个 step，却返回了 11 行。客户端看到的 8 行零，在 decode 的响应里就已经存在了。

这 11 行来自 decode 在请求结束时对本地缓存的一次读取。两个 worker 各有一张 expert 记录表，初始化时清零，每次 forward 只写入本次处理的 token。decode 接收了 prompt 的 KV cache，可以继续生成，但 prefill 为这 8 个位置选出的 expert ID 仍保存在 prefill 的表里。[缓存初始化与写入](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/state_capturer/base.py#L63-L118)

在实测的 SGLang v0.5.23 中，`get_topk` 按整个请求的序列长度取数。本例从位置 0 开始，读取区间是 `[0, 8 + 4 - 1)`，恰好覆盖 P1 到 P8 和 G1 到 G3。前 8 个位置没有写入这次 prompt 的 expert ID，实测读到的都是零。后 3 个位置由 decode 自己计算，所以能读到 G1 到 G3 的实际记录。[get_topk 的读取区间](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/state_capturer/base.py#L173-L188)

## 为什么从 decode 的末尾取

怎么拼这两份数据？直接把 prefill 的 8 行接在 decode 的 11 行前面，会让同一个 prompt 在结果里出现两次。反过来，如果规定一律跳过 decode 的前 8 行，又隐含了一个假设：decode 一定会返回 prompt 对应的位置。如果它只返回自己实际计算的 3 行，这一刀会把有效数据也切掉。

两种布局有一个共同点。请求生成了 4 个 token，第一个由 prefill 给出，剩下 3 个各需要一次 decode forward。因此无论 decode 返回 11 行还是 3 行，我们要的都是它**最后 3 行**。按行来写，本例的合并就是：

```python
prompt_routes = prefill_routes                 # P1 … P8
generated_routes = decode_routes[-3:]          # G1、G2、G3
routes = concatenate([prompt_routes, generated_routes])
```

实际响应里装的是 base64 编码的字节。补丁解码 prefill 的记录，用字节数除以请求的 prompt 行数（`prompt_tokens - routed_experts_start_len`），得到一行占多少字节。这样 router 不用预先知道模型有多少层、每层选几个 expert。随后用 `completion_tokens - 1` 算出需要保留的 decode 后缀长度，校验 decode 的数据长度符合上述两种布局之一，再截取后缀、拼接并重新编码。一个 batch 里如果同时有 8-token 和 23-token 的 prompt，就分别用各自的长度算边界。[merge_experts 实现](https://github.com/sgl-project/sglang/blob/494a59616d3acf44db9abafb98aa9df1ce244481/sgl-router/src/server/routes/chat/pd_metadata.rs#L33-L99)

只生成 1 个 token 时，prefill 已经完成了全部计算，decode 即使没有返回 expert 字段，也可以直接用 prefill 的记录。如果生成预算是 0，则要按非 PD 的读取区间再裁掉最后一个 prompt 位置。对于这里的 8-token 输入，返回 7 行。

最后我们用非 PD 的结果核对这个切分位置，逐 token 比较每层 expert ID 的顺序。图中的单条请求在修复前只有最后 3 行匹配，修复后 11 行全部匹配。混合长度 batch、不同读取起点和短生成的测试结果见 [PR #53355](https://github.com/sgl-project/sglang/pull/53355)。

## 引用本文

```bibtex
{% include citations/sglang-prompt-expert-ids.bib %}
```
