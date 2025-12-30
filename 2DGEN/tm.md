## 可能的冗余/待确认点
- `2DGEN/scrip/sample_and_export.py` 中 `grid_to_structure()` 使用 `coord_mask` 与 `atomic_vals > 0` 的并集；移除可能导致在原子通道偏小但坐标有效时丢失原子，目前保留为兜底逻辑。
