# v4 / D4 报告资料包说明

本资料包用于课程项目报告写作和结果追溯。建议先读 `01_report_brief/FINAL_EXPERIMENT_REPORT_LLM_BRIEF.md`，再按需要查看代码、实验配置和原始 JSON。最终报告的章节必须遵循项目说明 Section 9；写作底稿的第 11 节已按该格式排好。

## 最终保留模型

- 主模型 D4：`03_models/d4/round-0150.pt`
  - SHA-256：`3fe04112793ea8fd8fde7f72e7da40f4bb5e54f7ebe0a45f5f1b92f378ac3dee`
  - 基础权重来自 CNN n=8 训练的 `control-r2-round0150`；部署时使用一步碰撞 mask 和 D4 对称集成推理。
  - 参赛名为 `niulai`；可直接提交的单目录推理代码在 `02_code/full_project_source/agent_code/niulai/`。其中 `model.pt` 只保留与原 checkpoint 完全相同的在线网络权重。
- 对照模型 v4：`03_models/v4/model_a.pt`
  - SHA-256：`d5fe215b1f909149e1eefa2ee6ac00d7365e61343b333ba91b900e11e911d239`
  - 局部手工特征、MLP Dueling Double DQN，是稳定性对照和模型演化起点。

## 目录

- `00_official/`：项目说明 PDF 和训练指南。
- `01_report_brief/`：供组员或大模型起草报告的主线底稿。
- `02_code/full_project_source/`：完整项目源码快照，包括游戏框架、项目自研 agent、训练/评估工具和测试。
- `02_code/agent_code/`：为便于快速阅读而单独保留的 v4、CNN n=8、碰撞 mask、D4 对称推理最小依赖链。
- `03_models/`：最终保留的两个权重文件。
- `04_configs/`：课程训练、诊断和最终确认的预注册配置。
- `05_results/v4_baseline/`：v3/v4 关键原始评估 JSON。
- `05_results/curriculum/`：Task 1、Task 2、Task 3 peaceful、Task 3 coin、Task 4 的课程报告。
- `05_results/diagnostics/`：自杀、炸弹安全、碰撞 mask、reward shaping 等聚合诊断。
- `05_results/final_d4/`：D4 A/B、selection-free confirmation 和 10-seed 结果。
- `05_results/rejected_models/`：若干被淘汰路线的摘要，用于报告中的消融与失败分析。
- `06_tools/`：生成这些结果的主要训练、评估和单元测试脚本。
- `MANIFEST.sha256`：包内文件校验值。

## 使用原则

1. 报告中的数值优先引用 JSON，而不是从聊天记录抄写。
2. 区分 checkpoint 选择集与最终 confirmation seeds；不要把事后选峰值写成独立验证。
3. 不要把 D4 描述成重新训练的网络。D4 是固定 CNN 权重上的几何对称推理增强。
4. 如实报告失败路线、seed 方差和课程遗忘；叙述重点放在“问题—证据—改动—验证”，不要写成逐日流水账。
5. 本包没有收录逐步 trace、缓存、重复 checkpoint 和临时输出。完整源码不等于完整训练产物；聚合报告保留了主要指标。
6. 本包只整理本项目自身的模型演化、内部基线与官方环境评估材料；不要在最终报告中引入包外比较材料。

## 可复现性说明

代码按原项目包结构保存。D4 的推理依赖链为：

`model_a_cnn_n8_symmetry_ab` → `model_a_cnn_n8_league_train` → `model_a_cnn_n8_postbomb_movement_audit` → `model_a_cnn_n8_robust_bomb_audit`，同时依赖 `model_a_cnn_n8`、`model_a_dqn.features` 和 `model_a_v6.symmetry`。

`02_code/full_project_source/` 保留原始目录结构，可以作为代码审阅和复现实验的起点。D4 最终推理目录自包含；历史训练权重仍集中放在 `03_models/`，protocol/config 集中放在 `04_configs/`。运行特定历史实验时需要按配置恢复相应路径和环境变量。本资料包不包含大体积训练轨迹，因此不是整个工作目录的逐字节镜像。
