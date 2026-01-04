# twodgen 对齐计划（几何与图构建）

## 目标
对齐 2D 结构的几何字段与采样流形，补齐双图与 PBC wrap 信息，提升训练稳定性与采样合法性。

## 高优先级（必要）
### 1) uv_angle / z_norm / lattice_param 纳入前向与损失 ✅
- 范围：`twodgen/model/atom_transformer.py`、`twodgen/model/atom_denoiser.py`、`twodgen/common/atom_diffusion.py`
- 方案：
  - 将 `uv_angle/z_norm/lattice_param` 作为条件或额外扩散变量输入模型。
  - 增加回归头（或 v-pred）并加入 loss（训练期）。
  - 采样期与投影联动，保持输出在合法域。
- 验证：
  - 训练日志新增对应 loss 项；
  - 采样后字段分布与训练集对齐。
- 已验证：`twodgen/scrip/test/test_geom_graph.py`

### 2) 采样期几何投影（uv_angle / z_norm）✅
- 范围：`twodgen/model/atom_denoiser.py` 采样循环、`twodgen/scrip/sample_tokens.py` CLI
- 方案：
  - 新增 `--project-geometry` 开关。
  - 每步采样后投影：`uv_angle` 回单位圆，`z_norm` clip 到 `[-z_norm_clip, z_norm_clip]`。
  - 采样期同步更新 `lattice_param/t`，但不做投影。
- 说明：
  - 当前投影仅覆盖 `uv_angle/z_norm`；`lattice_param/t` 仅更新不投影。
- 验证：
  - 采样结果无越界；
  - 可对比开启/关闭的稳定性差异。
- 已验证：`twodgen/scrip/test/test_geom_graph.py`

### 3) 双图策略：kNN(d_xy) + kNN(d_3d) ✅
- 范围：`twodgen/model/atom_transformer.py` 图构建与注意力消息传递
- 方案：
  - 同时构建面内图与 3D 图；
  - 边类型区分（内层/跨层/重合），支持 edge_type embedding + gating。
  - 通过 `--dual-graph` 开关控制启用。
- 验证：
  - 图统计（边数/层内占比）符合预期；
  - 训练 loss 稳定性提升。
- 已验证：`twodgen/scrip/test/test_geom_graph.py`

### 4) wrap embedding（MIC 平移编码）✅
- 范围：`twodgen/model/atom_transformer.py` 邻居构建
- 方案：
  - MIC 返回的 (m,n) 映射到 `wrap_id∈[0..8]`；
  - wrap embedding 加入边特征（可用 `--wrap-embed-dim`）。
- 验证：
  - PBC 边的 wrap_id 分布合理；
  - 采样的周期一致性提高。
- 已验证：`twodgen/scrip/test/test_geom_graph.py`

### 5) t_head（厚度预测）✅
- 范围：`twodgen/model/atom_transformer.py` 输出头、`twodgen/common/atom_diffusion.py` loss
- 方案：
  - 预测 `t`（厚度）回归头；
  - 与 z_norm 一起加入损失。
- 验证：
  - `t` 的 MAE/分布对齐；
  - 采样厚度不塌缩。
- 已验证：`twodgen/scrip/test/test_geom_graph.py`

## 中优先级（补齐与配置）
- CLI 配置：
  - `--use-geometry-fields` / `--project-geometry` / `--dual-graph` / `--wrap-embed-dim`
- README 同步：
  - 训练与采样示例加入新开关与解释。

## 依赖与顺序建议
1. 几何字段前向 + loss（uv_angle/z_norm/lattice_param）
2. 采样几何投影
3. t_head（厚度闭环）
4. 双图 + edge_type
5. wrap embedding
