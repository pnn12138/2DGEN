# Adapter（多层条件注入）实现逻辑说明（供 Codex 落地）

> 目标：在你们当前 **2D Crystal / Flow Matching 生成模型**中，引入 **“单主干 + 每层轻量 FiLM/门控式 Adapter”**，显著增强 **化学式/组成条件（composition/counts）** 对生成的控制力，同时避免“双分支模型不对齐导致难训练”的风险。

---

## 0.5 当前工程的实际落地路径（务必按这个走）
你们已有 **单主干 + AdaLN/gating**（`twodgen/model/atom_transformer.py` 内 `AtomBlock.mod`）。
因此 **不改主干**，只补齐 **composition 条件编码** 与 **配置/脚本**。

**最小侵入方案（强烈建议）**
- 保持 `cond_vec` 为固定维度，不改 `train_tokens.py`/`sample_tokens.py` 的输入输出协议。
- 在模型内新增 `CompositionEncoder`，输入仍用 `counts_vector`（来自已有 batch），输出 `cond_comp`（shape `(B, D)`）。
- 用 `cond = cond_time + cond_comp (+ cond_mlp(cond_vec))` 的方式融合。

这样可以：
- 保留老 checkpoint / 脚本逻辑；
- 不动采样/训练 CLI；
- 只改模型文件 + 极少量配置。

---

## 1. 总体方案：单主干 + 每层 FiLM / Gate Adapter

### 1.1 为什么不做双分支
你担心的“训练不对齐难训练”通常来自：
- 双主干（两套 GNN/Transformer）各自 LN/统计不同
- 最后才融合导致梯度冲突

本方案避免该问题：
- **只有一个主干 denoiser**
- adapter 是 **每层一个小残差调制模块**
- 主干统计/LN 一致，训练更稳

---

## 2. 输入与条件 `c` 的设计（重点：composition 条件）

### 2.1 条件信息推荐组成
最少要包含（用于 composition 控制）：
- **元素种类 + 计数**（来自化学式）
- 总原子数 `N_total`
- （可选）元素摩尔分数 `fraction_k = count_k / N_total`

可选增强（你们已对齐的几何字段也可接入）：
- `t`（厚度或 vacuum 相关尺度）
- `lattice_param`（a,b,c 或 gram/cholesky 表示）
- `uv_angle`、`z_norm` 的全局统计（如 mean/std 或直接值）

### 2.2 不建议：直接用超长稀疏 counts_vector 作为唯一条件
原因：
- 长度大、稀疏、容易被网络忽略
- 对新元素组合泛化较差

更推荐：**DeepSets 风格的 composition pooling**（见 3.2）

---

## 3. ConditionEncoder：把化学式变成固定维度向量

### 3.1 输入格式（建议在 dataloader/collate 里准备）
每个样本提供：
- `elem_ids`: shape `[E]`（该材料含 E 种元素）
- `elem_counts`: shape `[E]`（对应计数）
- 可选 `geom_cond`: 例如 `[t, lattice_param..., uv_angle...]`

batch 化后：
- `elem_ids`: `[B, E_max]`（padding）
- `elem_counts`: `[B, E_max]`（padding 位置 count=0）
- `elem_mask`: `[B, E_max]`（True 表示有效元素位）

### 3.2 Composition pooling（推荐实现）
定义元素 embedding 表 `EmbZ`: `Z -> R^{d_c}`

对每个样本：
- `e_k = EmbZ(elem_ids[k])`  -> `[d_c]`
- `w_k = elem_counts[k]` 或 `sqrt(count)`（建议用 `sqrt` 稳定）
- `comp_vec = sum_k w_k * e_k`  -> `[d_c]`
- `frac_vec = sum_k (count_k/N_total) * e_k`（可选，与 comp_vec 拼接）

最终：
- `cond_raw = concat([comp_vec, frac_vec, N_total, geom_cond...])`
- `c = MLP(cond_raw) -> R^{d_model}`（与主干 hidden 同维）

### 3.3 输出
- `c`: `[B, d_model]`（图级条件向量）
- 后续会 broadcast 到节点维度 `[B, N, d_model]`

---

## 4. Adapter 结构：FiLM / Gate（推荐 FiLM）

### 4.1 FiLM 概念
对隐藏特征 `h` 进行仿射调制：

```
h' = h * (1 + gamma) + beta
```

其中 `gamma, beta` 由条件向量 `c` 生成。

### 4.2 生成 gamma/beta 的方式（稳定优先）
推荐每层一个轻量 “FiLMGen”：
- 输入：`c` `[B, d_model]`
- 输出：`gamma_l, beta_l` `[B, d_model]`

实现方式：
- `u = SiLU(Linear_shared(c))`（shared across layers，减少参数）
- `gamma_l = Linear_gamma_l(u)`（per-layer small linear）
- `beta_l  = Linear_beta_l(u)`（per-layer small linear）

**关键初始化（非常重要）**：
- `Linear_gamma_l` 和 `Linear_beta_l` 的权重/偏置初始化为 0
  => 初始时 `gamma=0, beta=0`，Adapter 等价于“不存在”，训练更稳。

### 4.3 broadcast 方式
- `gamma_l`: `[B, d_model]` -> `[B, 1, d_model]` -> broadcast 到 `[B, N, d_model]`
- 同理 `beta_l`

---

## 5. Adapter 插入位置（单主干最稳策略）

### 5.1 推荐：每个 block 开头插一次（最省心）
假设主干是 Transformer/GNN block（含 Attention/MessagePassing + FFN）：

**建议默认：block 前一次 FiLM（然后走原 block）**
```text
h = FiLM(h, gamma_l, beta_l)         # 条件调制
h = Block(h, edge_index, edge_attr)  # 原本的 attention/MP + FFN
```

优点：
- 最小侵入
- 不需要改 block 内部结构
- 不改变 LN/残差结构太多，稳定

### 5.2 进阶：Attention 前 + FFN 前各一次（可选）
如果你们发现条件仍然偏弱，再升级为：
```text
h = h + Attn(LN(FiLM(h)))
h = h + FFN(LN(FiLM(h)))
```
但建议先做 5.1 的 MVP。

---

## 6. 在 Flow Matching forward 中如何接入（关键流程）

### 6.1 forward 输入（示例）
- `x_t`: 当前状态（节点/晶格/其他）
- `t`: 时间标量或 embedding
- `cond_dict`: 包含 elem_ids/counts/geom_cond 等

### 6.2 forward 流程（伪代码）
1) `c = ConditionEncoder(cond_dict)`  -> `[B, d_model]`
2) `h = NodeEmbed(x_t, t)` -> `[B, N, d_model]`
3) 对每一层 `l in [1..L]`：
   - `gamma_l, beta_l = FiLMGen_l(c)` -> `[B, d_model]`
   - `h = FiLMApply(h, gamma_l, beta_l)` -> `[B, N, d_model]`
   - `h = Block_l(h, graph)` -> `[B, N, d_model]`
4) `v = OutputHead(h, ...)` -> 向量场预测（与现有实现一致）

---

## 7. 条件 dropout（为 CFG 做准备，可选但强烈建议顺手加）
> CFG 的训练方式就是“随机丢条件”，让模型学到 conditional 和 unconditional 两种行为。

实现：
- 在训练时，以概率 `p_drop` 将 `cond_dict` 替换为 “空条件”
  - elem_counts 全 0
  - geom_cond 全 0
  - 或用 special token embedding

注意：
- 推理时如果不做 CFG，仍然可以不启用 drop
- 但建议加上这个开关，为后续 CFG 打基础

---

## 8. 配置与开关（建议做成最少但可控）

建议新增配置项：
- `model.use_adapter: bool`
- `model.adapter_type: film | gate`（默认 film）
- `model.adapter_shared_dim: int`（shared hidden，例如 d_model//2）
- `train.cond_drop_prob: float`（默认 0.1~0.2）
- `adapter.init_zero: bool`（默认 True，强制 gamma/beta 初值为 0）

并保留 ablation：
- `--use_adapter` 开/关
- `--cond_drop_prob` 扫描
- `--film_pos=pre_block|pre_attn_ffn`（可选）

---

## 9. 你需要产出的“可验证结果”（Adapter 是否真的起作用）

### 9.1 必做指标（composition 控制）
- 完全匹配率：元素集合 + 计数完全一致（给定化学式条件时）
- 元素集合命中率：只看元素种类是否一致
- counts 误差：`L1`/`L2` 差值分布
- 条件敏感性：同 seed 不同条件，生成分布差异显著

### 9.2 建议消融
- baseline：无 adapter
- + adapter：只加 adapter（FiLM）
- + adapter + cond_drop（后续可加 CFG 推理）
输出对比表（论文最需要）

---

## 10. 最小单元测试/自检（强烈建议写）

1) **等价性测试**
- adapter 关闭 vs adapter 开启但 gamma/beta=0 => 输出应几乎一致

2) **shape 测试**
- 不同 N（变长图）能正确 broadcast gamma/beta
- batch padding 的 elem_ids/counts 不影响 c（mask 正确）

3) **梯度测试**
- 反传时 adapter 参数有梯度，主干也有梯度
- cond_drop 时模型无 NaN

---

## 11. 后续扩展（留接口）
Adapter 一旦做好，后续很自然接入：
- XRD embedding（把谱编码成向量拼到 `cond_raw`）
- 目标性质（bandgap/ehull）作为额外条件
- CFG 推理（使用 cond/uncond 的向量场线性组合）

---

# 附：模块清单（建议你在代码里新增/改动的组件）

**新增**
- `CompositionEncoder`（输入 counts_vector -> comp_vec -> cond_comp）
- （可选）`FiLMGenShared`（shared MLP）
- （可选）`FiLMGenPerLayer`（每层 Linear_gamma/Linear_beta）
- config/flags（use_comp_encoder, comp_embed_dim, comp_pool_mode, comp_use_frac）

**改动**
- `AtomTransformer.forward(...)`：新增 `cond_comp` 路径；`cond = cond_time + cond_comp (+ cond_mlp(cond_vec))`
- `AtomTransformer.__init__`：新增元素 embedding + pooling + MLP
- `train_tokens.py`/`sample_tokens.py`：仅新增配置保存/加载（不改 cond_vec I/O）

---

## MVP 建议（最小可落地版本）
如果你希望最快见效，按这个最小集合实现即可：
1) **CompositionEncoder（模型内部）**：`counts_vector` -> `comp_vec` -> `cond_comp`
2) **保持现有 AdaLN/gating**（不改主干 block）
3) `use_comp_encoder` 开关 + 简单 ablation
4) 输出 composition match 指标（现成）

做完这一步，通常就能明显提升“给定化学式条件时的匹配度”。

---

# 具体改造方案（按文件落地）

## A) 模型侧：新增 CompositionEncoder（不改主干）
目标：不改变 `cond_vec` 协议，仅增强 composition 表达。

**位置**：`twodgen/model/atom_transformer.py`

**新增配置（AtomTransformerConfig）**
- `use_comp_encoder: bool = False`
- `comp_embed_dim: int = 64`（元素 embedding 维）
- `comp_pool_mode: str = "count"`（`count | sqrt | frac`）
- `comp_use_frac: bool = True`（是否拼接 fraction pooling）

**新增模块（在 __init__ 中）**
- `self.comp_embed = nn.Embedding(num_elements + 1, comp_embed_dim, padding_idx=0)`
- `self.comp_mlp = nn.Sequential(Linear(...), SiLU(), Linear(...))`
  输入维度建议：
  - `comp_vec` (D)
  - `frac_vec` (D, 可选)
  - `N_total` (1)
  合计 `(D or 2D) + 1 -> embed_dim`

**新增方法（伪代码）**
```python
def _encode_composition(self, counts_vector: torch.Tensor) -> torch.Tensor:
    # counts_vector: (B, num_elements)
    counts = counts_vector.float()
    total = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    elem_ids = torch.arange(1, counts.size(-1) + 1, device=counts.device)
    emb = self.comp_embed(elem_ids)  # (E, D)
    if self.cfg.comp_pool_mode == "sqrt":
        weights = counts.sqrt()
    elif self.cfg.comp_pool_mode == "frac":
        weights = counts / total
    else:
        weights = counts
    comp_vec = weights @ emb  # (B, D)
    parts = [comp_vec]
    if self.cfg.comp_use_frac:
        frac_vec = (counts / total) @ emb
        parts.append(frac_vec)
    parts.append(total)  # N_total
    return self.comp_mlp(torch.cat(parts, dim=-1))
```

**在 forward 中融合**
- 现有：`cond = cond_time + cond_mlp(cond_vec)`
- 改为（伪代码）：
```python
cond = cond_time
if cond_vec is not None and self.cond_mlp is not None:
    cond = cond + self.cond_mlp(cond_vec)
if self.cfg.use_comp_encoder and counts_vector is not None:
    cond = cond + self._encode_composition(counts_vector)
```
说明：`counts_vector` 可直接用现有 batch 的 `counts_vector`（训练时来自 dataset，采样时来自 cond-npz）。

## B) 训练/采样脚本：只加 config，不改协议
**位置**：`twodgen/scrip/train_tokens.py`, `twodgen/scrip/sample_tokens.py`

**动作**
- CLI 添加 `--use-comp-encoder/--comp-embed-dim/--comp-pool-mode/--comp-use-frac`（仅保存进 config）。
- 保存到 checkpoint 的 `cond_config`/`model_config` 里。
- 读取 checkpoint 后构建 `AtomTransformerConfig` 时填回这些字段。

**重点**：`cond_vec` 的构建逻辑不变，`counts_vector` 仍是 batch 字段（train）或 `cond-npz` 字段（sample）。

## C) Dataset/Collate：不改
`counts_vector` 已存在于 `C2DBAtomDataset`/`C2DBTokenNPZDataset`，无需新增变长 `elem_ids`。

## D) 可选：zero-init（增强稳定）
如果需要和文档完全一致：
- 将 `AtomBlock.mod` 的最后一层线性权重/偏置初始化为 0（zero-init）。
- 但这不是必要条件，属于可选稳定性增强。

---

# 验证与消融（适配当前脚本）
1) `--use-comp-encoder` on/off 比较
2) `comp_pool_mode=count|sqrt|frac` 扫描
3) 组合 `cond_drop_prob`（已有）
4) 输出 `evaluate/eval_samples.py` 的 composition match 指标（已有）

---

# 兼容性与风险
- **不破坏**旧 checkpoint 与脚本：`cond_vec` 协议不变。
- 如果 `counts_vector` 缺失（如历史 npz），`use_comp_encoder=True` 会报错；需保证数据包含 `counts_vector`。

---

# 上线前必处理的关键风险与补丁

## 风险 A：counts_vector 的“元素索引含义”必须对齐
当前伪代码用 `elem_ids = arange(1..E)` + `Embedding(num_elements+1, padding_idx=0)`，隐含假设：
`counts_vector[k]` 对应 **原子序数 Z = k+1**。

但数据集常见两种编码：\n
- `counts_vector[Z]` 与原子序数 Z 对齐（Z=1..118）\n
- `counts_vector[idx]` 与“训练用元素表内部索引”对齐（idx=0..M-1）\n

如果是第 2 种，`arange(1..E)` 会 **错位**，composition 条件变成“错元素的加权和”。\n
**建议**：把“counts index -> 元素 Z”的映射固化为常量 buffer，由 dataset 定义一次，模型只使用该映射。

**实现建议（示意）**\n
- dataset 输出 `element_ids_tensor`（shape `(E,)`，取值为原子序数）\n
- 模型 `register_buffer("elem_ids", element_ids_tensor)`，pooling 用 `comp_embed(elem_ids)`，而不是 `arange(1..E)`。

## 风险 B：cond 直接相加可能尺度不匹配（互相淹没）
当前融合是 `cond = cond_time + cond_vec_proj + cond_comp`，三者尺度不同会导致：\n
- `cond_time` 太大：composition 被淹没，条件仍弱\n
- `cond_comp` 太大：训练早期不稳\n

**强烈建议**：加入可学习标量门控，并 **w_comp=0 起步**（兼容旧 ckpt）：
```python
cond = w_time * cond_time + w_vec * cond_vec_proj + w_comp * cond_comp
```
初始化：`w_time=1, w_vec=1, w_comp=0`。\n
这样加载旧 checkpoint 时新增路径是严格 no-op，再逐渐学会用 composition。

## 风险 C：CFG/cond-drop 必须同时 drop composition
如果 `cond_drop_prob` 只 drop `cond_vec`，而 `cond_comp` 仍由 `counts_vector` 计算，\n
所谓“unconditional”其实仍看到了化学式，CFG 会失效。

**建议**：cond-drop 触发时，同时把 `counts_vector` 置零或直接 `cond_comp=0`。

---

# 需要同步补的 3 个小改动（优先级高）
1) **w_comp 标量门控（初值=0）**：兼容旧 ckpt + 防止尺度不稳。\n
2) **counts 映射固化**：避免元素错位导致“隐蔽灾难”。\n
3) **cond-drop 同时作用于 comp**：保证 CFG 可用、无条件分支真实。\n

---

# 预期效果与失败信号
**正向变化**\n
- composition match rate 上升（先是元素集合命中率，再是计数完全一致）\n
- 同 seed 换化学式条件，生成分布明显变化\n
- 训练稳定性基本不变（主干没动，w_comp 从 0 起步）\n

**异常信号**\n
- match 率不升反降或出现“错元素偏好”：优先检查 counts→元素映射\n
- loss 早期剧烈波动/NaN：优先检查 cond 融合尺度（建议上 w_comp）\n
- CFG 没效果：检查 cond-drop 是否包含 counts/cond_comp\n
