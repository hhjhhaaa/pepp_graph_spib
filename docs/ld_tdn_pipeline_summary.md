# LD-TDN 课题 Pipeline 总结

## 1. 课题定位

本课题面向 **孔道受限聚合物输运**，目标是从 ps 级局部 MLFF-MD 轨迹中学习可解释的动态输运描述符。重点不是用机器学习力场暴力外推长时间、大体系扩散，而是：

```text
DFT/AIMD 锚定的 MLFF
→ ps 级局部高保真轨迹
→ segment-centered short trajectory windows
→ local dynamic transport descriptor z_i
→ region-wise pooling
→ physics-informed transport factors
→ D_eff / residence / escape / accessibility / wall trapping / axial mobility
```

当前 downstream 模型只关注 transport、diffusion、residence、escape、accessibility、wall trapping 和 axial mobility，不做化学事件建模。

## 2. 整体方法学主线

完整 pipeline 是：

```text
初始结构生成
        ↓
DFT/AIMD 局部标注与 MLFF 锚定
        ↓
MLFF-MD 生成 ps 级局部轨迹
        ↓
局部短轨迹窗口化
        ↓
LD-TDN 学习 dynamic transport descriptor
        ↓
region-wise pooling 得到孔道/体系级描述符
        ↓
physics-informed transport head 输出输运因子
        ↓
descriptor table + LASSO / sparse regression 解释
        ↓
服务孔道输运机制分析与实验变量筛选
```

核心逻辑是：

```text
短轨迹不直接给 D_eff；
短轨迹先给局部动态描述符；
局部描述符再通过区域聚合和物理约束模型连接到孔道输运性质。
```

## 3. 初始结构与 MLFF 锚定

Packmol、EMC 或 classical force field 可以用于生成初始 packing、去除坏接触，并提供 PE / PP / PS 与 SiO2 / SiOH 局部构型起点。classical MD 不作为最终动力学标签来源，也不作为机理结论依据。

DFT/AIMD 标注应覆盖：

```text
bulk polymer:
  PE 局部堆积
  PP 局部堆积
  PE/PP 接触
  PE/PS 接触
  PP/PS 接触
  PS 苯环侧基取向和芳香局部构象
  高密度 packing 区
  低密度 / free-volume 区
  二面角转换构型

confined / interface environment:
  polymer-SiO2 接触
  polymer-SiOH 接触
  silanol-rich wall
  siloxane-rich wall
  near-wall slow mobility region
  pore-entry conformation
  crowded polymer-wall contact
```

MLFF 的任务是得到一个对 PE/PP/PS 及 polymer-SiO2/SiOH 局部环境可靠的专用力场，用于产生 ps 级高保真局部轨迹。

## 4. MLFF-MD 轨迹与窗口化

MLFF-MD 主要用于生成几千原子以内、ps 级局部动力学轨迹。轨迹需要保存：

```text
coordinates
velocity or displacement
chain ID
segment ID
polymer type: PE / PP / PS
wall distance
local density
local composition
contact information
dihedral / orientation proxy
free-volume proxy
pore coordinate / radial bin / axial bin
```

每条轨迹切成 segment-centered local windows。每个样本是：

```text
LocalWindowSample:
  system_id
  center_id
  center_type
  feature_sequence [T, F]
  optional local graph_sequence
  condition
  local_labels
  system_targets
  target_mask
  metadata
```

`feature_sequence [T, F]` 是默认主输入，来自历史窗口，不能包含未来标签。

## 5. 显式条件变量

链长、重复单元数、组成、孔径和壁面化学是理论上可控的体系变量，必须显式输入 condition，不能指望模型从局部邻域自动推断。

```text
density
temperature
PE_fraction
PP_fraction
PS_fraction
PE_chain_length
PP_chain_length
PS_chain_length
PE_repeat_units
PP_repeat_units
PS_repeat_units
mean_chain_length
chain_length_polydispersity
pore_diameter
pore_length
silanol_density
wall_type_id
surface_hydroxylation_fraction
```

Dummy v1 是 PE/PP-only；PS 字段是 schema placeholder，在真实 PS preprocessing 实现前置零。

## 6. LD-TDN 模型架构

默认路径：

```text
feature_sequence [T, F]
        ↓
GRU / TCN temporal encoder
        ↓
condition encoder
        ↓
concat temporal_h + condition_h
        ↓
variational / predictive bottleneck
        ↓
z_i local dynamic transport descriptor
        ↓
local future-dynamics heads
```

可选 local ego-GNN 路径：

```text
local graph_sequence
        ↓
small local GNN per frame
        ↓
GRU / TCN temporal encoder
        ↓
z_i
```

禁止构建 full-system atomistic GNN。

## 7. 局部未来动力学任务

分类任务：

```text
mobility_class
contact_class
residence_class
escape_class
relax_class
```

回归任务：

```text
future_disp_parallel
future_disp_radial
future_disp_norm
short_msd_parallel
short_msd_radial
contact_survival
wall_contact_survival
free_volume_opening
```

这些任务来自未来短窗口，用于约束 `z_i` 保留真实动力学信息。训练损失包括 CE、Gaussian NLL / regression loss、MSE 和 bottleneck KL loss。

## 8. Region-wise Pooling

大量 local windows 的 `z_i` 通过 region-wise pooling 聚合成 pore/system-level transport descriptors。聚合维度包括：

```text
system_id
radial bin
axial bin
near-wall / pore-center
pore-mouth / inner-pore
center segment type
PE / PP / PS local environment
```

聚合统计包括：

```text
mean_z
var_z
fraction_fast
fraction_slow
fraction_persistent_contact
fraction_wall_resident
fraction_escape_ready
mean_free_volume_proxy
mean_wall_distance
mean_local_density
mean_polymer_wall_contact_fraction
mean_local_PE_fraction
mean_local_PP_fraction
mean_local_PS_fraction
radial_bin_fraction
axial_bin_fraction
```

## 9. Physics-Informed Transport Head

系统级 head 先预测具有物理意义的中间变量：

```text
D_local > 0
P_entry in [0, 1]
C_axis in [0, 1]
tau_wall > 0
tau_move > 0
P_access in [0, 1]
```

派生：

```text
transport_score = P_entry * C_axis / (1 + tau_wall / tau_move)
D_eff = D_local * transport_score
wall_residence_fraction = tau_wall / (tau_wall + tau_move)
```

输出解释重点：

```text
为什么某个体系输运慢？
是 D_local 小？
是 P_entry 低？
是 C_axis 差？
还是 tau_wall / tau_move 太大？
```

## 10. Descriptor Table 与稀疏解释

每个 system-level row 包含：

```text
condition variables
mean_z1 ... mean_zK
var_z1 ... var_zK
fraction_fast / fraction_slow
wall and pore pooled descriptors
D_local / P_entry / C_axis / tau_wall / tau_move / P_access
wall_residence_fraction / transport_score / D_eff
available target_* columns
```

之后使用 LASSO / sparse regression 将神经描述符和 transport outputs 蒸馏成少数显式物理描述符。

## 11. Validation / Ablation

当前 ablation 设计：

```text
condition_only
static_features_only
shuffled_history
no_condition
no_chain_length
no_composition
no_wall_features
descriptor_time_series_full
optional_local_gnn
```

关键验证点：

```text
shuffled_history 变差：模型利用时间顺序
no_chain_length 变差：链长 / repeat unit 显式输入有必要
no_composition 变差：composition/contact 信息有用
no_wall_features 变差：孔壁和孔道特征控制受限输运
optional_local_gnn 若优于 descriptor-only：局部空间图结构有额外贡献
```

## 12. 当前代码链路

```text
数据 schema:
  src/pepp_graph_spib/data/sample.py

dummy 数据生成:
  src/pepp_graph_spib/data/dummy.py
  scripts/make_dummy_graph_data.py

特征序列:
  src/pepp_graph_spib/features/segment_features.py

batch collate:
  src/pepp_graph_spib/data/collate.py

dataset / ablation:
  src/pepp_graph_spib/data/dataset.py

时间编码器:
  src/pepp_graph_spib/models/encoders/timeseries_encoder.py

可选 local GNN:
  src/pepp_graph_spib/models/encoders/local_gnn_encoder.py

bottleneck:
  src/pepp_graph_spib/models/bottleneck.py

局部动力学 heads:
  src/pepp_graph_spib/models/heads/local_dynamics_heads.py

LD-TDN 主模型:
  src/pepp_graph_spib/models/ld_tdn.py

local descriptor 训练:
  scripts/train_local_descriptor.py
  src/pepp_graph_spib/training/common.py

region pooling:
  src/pepp_graph_spib/models/pooling/region_pooling.py

physics transport head:
  src/pepp_graph_spib/models/heads/physics_transport_head.py

transport head 训练:
  scripts/train_transport_head.py

descriptor table:
  scripts/export_descriptor_table.py
  src/pepp_graph_spib/symbolic/descriptor_table.py

LASSO:
  scripts/run_symbolic_lasso.py

配置:
  configs/model_descriptor_only.yaml
  configs/model_local_gnn.yaml
  configs/model_pore_transport.yaml
```

## 13. 一句话版

当前模型是一个面向孔道受限聚合物输运的短轨迹动态描述符框架：它用 ps 级局部 MLFF-MD 轨迹学习 segment-level `z_i`，再通过 region-wise pooling 和 physics-informed transport head 解释 `D_eff`、residence、escape、accessibility、wall trapping 和 axial connectivity。
