# Bomberman 最终实验报告写作底稿

> 用途：将本文件交给大语言模型，据此撰写课程项目的最终实验报告。
> 本文件是事实底稿，不是最终正文。除非重新读取原始 JSON，否则不要修改这里的数值、模型身份或实验结论。

## 0. 给报告写作者的硬性要求

1. 最终主模型固定为 **D4**；保留 **v4** 作为第二模型和安全基线。不要再从历史 checkpoint 中重新排名。
2. 正文只讨论本项目自研的 D4、v4，以及课程规定的官方内置对手。不要引入其他模型来源、权重来源或额外参赛者。
3. 语气要直接、自然，像学生在解释自己做过的实验。多使用“我们观察到”“结果表明”“因此保留/否决了该方案”，少用空泛总结。
4. 不要写成防御性声明。不要反复使用“需要强调的是”“这并不意味着”“尽管存在局限”“值得注意的是”“综上所述”等模板句。
5. 不要把报告写成实验流水账。主线应是：发现 v4 的结构性瓶颈 → 设计全图 CNN+n=8 路线 → 课程训练与失败诊断 → 用低介入推理修正 → D4 得到独立确认。
6. 失败实验要简洁、诚实地写成帮助收敛设计空间的证据，不要为每一次失败道歉，也不要把所有试验堆进正文。
7. 不得声称某一个组件被单独证明是全部增益的来源。CNN、全图状态、n=8、课程训练、reward、mask 和 D4 推理共同构成最终系统；其中只有单独做过对照的变量才能作因果表述。
8. 数字必须带实验上下文。不同 seeds、对手分布或协议下的绝对分数不能直接相减；可以比较行为画像，但不能伪装成配对显著性。
9. 不要提及写作工具、提示词或“由 AI 生成”。不要使用过度整齐的三段式套话。
10. 建议全文使用第一人称复数“我们”，方法部分准确，讨论部分有判断，不要把每句话都写成保守的可能性陈述。

## 1. 最终模型决策

### 1.1 主模型：D4

D4 是当前冻结的最强内部 agent。它不是八个独立网络，也没有继续训练新权重。其完整定义是：

```text
control-r2-round0150 冻结 CNN 权重
+ 训练阶段已经使用的合法/生存与一步碰撞一致性 mask
+ 测试时对同一状态构造正方形的 8 个 D4 对称视图
+ 同一 CNN 对 8 个视图一次批量前向
+ 将六动作 Q 值映射回原坐标并取平均
+ 在统一 mask 后选择最大 Q 动作
```

冻结权重：

```text
experiments/checkpoints/model-a-cnn-n8-opponent-league-ab-s162000/control/r2/round-0150.pt
SHA-256: 3fe04112793ea8fd8fde7f72e7da40f4bb5e54f7ebe0a45f5f1b92f378ac3dee
```

推理实现：

```text
agent_code/model_a_cnn_n8_symmetry_ab/   # 原始 D4 评估实现
agent_code/niulai/                       # 最终参赛目录，同一策略的独立打包版本
```

checkpoint 元数据：`architecture=model-a-cnn-n8-full33-v1`、`n_step=8`、`stage=task4`、
`arm=control`、`replica=r2`、`stage_rounds=150`、`epsilon=0.05`。网络参数量为 `367,863`。

### 1.2 保留模型：v4

v4 是紧凑、低延迟、低自杀的安全基线。保留它的意义不是继续训练，而是提供结构不同、行为更保守的第二策略，
同时作为最终报告中判断新路线是否真正改善的参照。

```text
agent_code/model_a_dqn/model_a.pt
SHA-256: d5fe215b1f909149e1eefa2ee6ac00d7365e61343b333ba91b900e11e911d239
```

checkpoint 记录累计 `10,300` 训练局、`772,588` 环境步、`173,200` 次梯度更新和 `epsilon=0.05`；
网络参数量为 `48,151`。

### 1.3 冻结规则

- 不覆盖以上两个 checkpoint。
- 不把历史上某个单次高分 checkpoint 重新提拔为最终模型。
- 其他 checkpoint 和日志只作为实验复现材料保留，不作为最终提交候选。
- 报告中将 D4 称为“最终模型”或“主模型”，将 v4 称为“保留基线”或“安全基线”。

## 2. 研究问题与模型演化

### 2.1 研究问题

项目的核心不是单纯提高击杀数，而是在官方计分下学习一个能收集金币、炸箱、存活并在需要时攻击的统一策略。
早期 v4 已经具备稳定的危险规避和炸弹使用能力，但后续诊断暴露了三个瓶颈：

1. **视野瓶颈**：v4 只编码以自己为中心的 `7×7` 局部区域，远处金币、对手和全局通路不可见。
2. **时间归因瓶颈**：一步 TD 很难把“放弹—封路—数步后产生结果”的收益传回早期动作。
3. **行为局部最优**：v4 在安全状态下具有稳定的 WAIT 偏好。它自杀少，但资源竞争和主动交互不足。

实验目标因此从“继续给 v4 加攻击奖励”转为：使用完整棋盘状态和足够长的 temporal horizon，让网络自己学习
资源、移动和攻击之间的关系，再用严格验证控制自杀和无效动作。

### 2.2 v4 是怎样得到的

v4 不是一次训练直接得到的，而是从一个局部 Dueling Double DQN 逐步修正出来。报告中可以把这段写成第一条
清晰的模型演化链：

```text
初始 Model A
  → 修正爆炸语义并加入基础安全 mask
  → v3 时间感知生存 mask
  → v4 进攻相关全局特征 + 300 局混合对手微调
```

1. **初始 Model A**：使用 `4×7×7` 局部特征、Dueling Double DQN、uniform replay 和一步 TD。先在
   classic 场景训练 5,000 局，再对规则对手训练 5,000 局，形成后来 v4 的 10,000 局主干权重。
2. **爆炸语义和基础 mask 修正**：工程审计发现，官方爆炸会穿过木箱、只被石墙阻挡。修复危险图后，v2 在
   五个 seeds 上得到 `2.376/局`，但自杀仍为 `32.6/100局`，没有通过安全门。
3. **v3 时间感知生存 mask**：不修改网络权重，改用时空搜索判断动作能否撑过所有已知爆炸，并要求放弹后
   存在带时间余量的逃生路线。v3 在五个 seeds 上由 `2.376` 提升到 `2.838/局`，自杀由 `32.6` 降到
   `17.0/100局`。更严格的“双出口/两步逃离”版本虽然更安全，却明显压制炸箱和得分，因此被淘汰。
4. **v4 进攻特征**：保留 v3 安全层，在原有 4 个全局标量后增加最近对手距离、对手受限程度、当前位置安全
   放弹能否直接覆盖对手。新增输入列以零初始化，保证微调前 Q 值不变；随后使用官方内置混合对手训练 300 局。
   checkpoint 因而记录总计 `10,300` 局。

v4 的正式配对结果如下：

| 配置 | v3 score/round | v4 score/round | v3 suicides/100 | v4 suicides/100 |
|---|---:|---:|---:|---:|
| mixed official roster | 2.770 | **4.073** | 12.7 | **10.3** |
| three rule-based | 2.817 | **2.997** | 18.0 | **13.7** |

在 mixed 配置中，v4 相对 v3 的得分提高约 `47.1%`、击杀提高约 `86.5%`；在三个 rule-based 配置中，
得分提高约 `6.4%`且自杀降低约`24.1%`。因此 v4 被冻结为当时最可靠的主模型。

### 2.3 从 v4 到 D4 的逐步路线

这里必须写清一个重要事实：**D4 在研究思路上继承 v4 的诊断结论，但 CNN 权重不是从 v4 权重继续训练的。**
我们先尝试低风险地改进 v4；确认局部表示和一步 TD 已成为瓶颈后，才从随机初始化建立全图 CNN。完整路线是：

1. **冻结 v4 并做局部改进。** 依次测试奖励对齐、战术特征、planner residual、训练时对称增强、bounded
   residual、动作覆盖和搜索/蒸馏路线。它们或者改变成熟 Q 排序、或者只在小样本改善、或者提高攻击但损失总分，
   均未通过独立门槛。
2. **复训 exact-v4 课程。** 使用同一 v4 架构从 Task1 开始按课程训练。Task2 的 `source-r2` 一度成为强候选，
   但继续 Task3/Task4 后出现遗忘和不稳定，最终没有形成优于冻结 v4 的完整模型。这证明问题不只是训练顺序。
3. **诊断 temporal credit。** 攻击事件在 replay 中极少，四步窗口又不足以覆盖“放弹到结果”的延迟。对照审计
   支持 `n=8`：它覆盖已观察到的有效 BOMB 链，而 `n=16/32` 没有增加有效覆盖。
4. **建立独立全图 CNN。** 从随机权重训练 11 通道 `33×33` CNN，保留 Double/Dueling DQN、uniform replay
   和安全 mask，使用有界 shaping，并严格分开 Task1、Task2、Task3-P、Task3-C、Task4。
5. **选择中间 checkpoint。** 三个 replicas 共同使用预注册 milestones 和 pooled early-near-best 规则，最终
   从 15 个课程冠军中选择 `task4-r3-round0800`，而不是默认使用训练最末端权重。
6. **定位并修正 CNN 自杀链。** 轨迹审计表明多数自杀发生在放弹后的连续移动冲突。一步碰撞一致性 mask 只
   修改极少数高风险动作，主要作用是降低 invalid，而不是替网络规划攻击。
7. **短剂量 Task4 control 续训。** 从 CNN 父 checkpoint 做受控末段训练，通过预注册 checkpoint 选择保留
   `control-r2-round0150`。更复杂的训练对手分布没有超过这个简单 control。
8. **D4 测试时对称平均。** 冻结 `control-r2-round0150`，对同一 CNN 的 8 个旋转/镜像视图做动作对齐后的
   Q 平均。s170 A/B 和 s171 selection-free confirmation 连续通过，最终形成 D4。

因此，报告中的演化关系应画成：

```text
局部 Model A → v2 → v3 → v4（保留基线）
                         │
                         ├─ 多条增量改进路线被淘汰
                         └─ 诊断出局部视野与时间归因瓶颈
                              → 随机初始化 full-board CNN+n=8
                              → Task1/2/3-P/3-C/4 课程
                              → task4-r3-round0800
                              → collision-consistent mask
                              → control-r2-round0150
                              → D4 对称推理（最终主模型）
```

### 2.4 具有代表性的淘汰模型

不需要在最终正文逐个展开，但 Methods 或 Experiments and Results 中应列举若干模型，证明选择是经过系统比较
得出的。推荐保留以下代表：

| 模型/路线 | 主要改动 | 淘汰原因 |
|---|---|---|
| v3-strict | 更严格的双出口/两步逃离炸弹门 | 自杀降低，但严重压制炸箱和总分 |
| v5 reward alignment | 提高击杀奖励并加重自杀惩罚 | mixed 得分比 v4 下降约32.3%，击杀下降约52.6% |
| v4.1 tactical lookahead | 六步空间特征和困杀特征，全网微调 | Q 值整体漂移；rule与mixed均低于v4，自杀上升 |
| v4.2a planner residual | 冻结 v4 主干，只训练小型规划 residual | 最高均分只提高约0.8%，invalid和自杀明显恶化 |
| v4.2b D4 augmentation | 在成熟 v4 上进行八重训练时对称增强 | 最保守版本得分仍下降44%，说明晚期强制等变破坏已有表示 |
| v6 residual family | 冻结 v4 加 bounded residual，并做多 seed/ensemble | 小样本有增益，但 held-out 或逐 seed 稳定门失败 |
| v7/v8 | 推理动作覆盖或 rollout 蒸馏 | 覆盖过强、收益不稳定，未超过冻结 v4 |
| v9 full-board pilot | 较早的大型全图 CNN 路线 | 训练后仍明显低于 v4，未在预算内收敛 |
| v10 search/student | 搜索、history-aware gate 与 student head | 工程复杂度高，Task1信号没有可靠迁移到完整 classic |
| exact-v4 Task3/4 descendants | 从课程 Task2 候选继续训练 | 出现遗忘、seed方差或最终总分回撤，没有形成统一冠军 |
| CNN safety replay | batch 中固定加入 kill/self-kill 链 | 自杀只小幅改善，总分和击杀明显下降 |
| productive-BOMB | 过滤当前射程内无木箱/敌人的炸弹 | 小样本通过，独立扩大确认中由3.460降至3.165 |

这些模型不能被写成“无意义尝试”。它们分别排除了单纯改奖励、晚期对称增强、强动作覆盖、复杂训练分布和
稀有事件过采样等路线，使最终设计逐渐收敛到全图表示、n=8、受控课程选择和测试时对称平均。

## 3. 两个模型的结构

### 3.1 v4：局部 MLP Dueling Double DQN

v4 输入包括：

- `4×7×7` 局部张量：静态墙/木箱、危险时间、炸弹计时、金币与对手；
- 7 个全局标量：可放弹、当前危险、剩余金币、邻近木箱或可放弹、最近对手距离、对手受限程度、当前放弹能否直接覆盖对手。

网络结构：

```text
4×7×7 + 7 scalars
→ Flatten → Linear(203,128) → ReLU
→ Linear(128,96) → ReLU
├→ Value:     96 → 48 → 1
└→ Advantage: 96 → 48 → 6
```

训练使用一步 Double DQN、Dueling head、uniform replay、Smooth-L1、`gamma=0.99`、batch 64、
learning rate `3e-4`。动作空间为 `UP/RIGHT/DOWN/LEFT/WAIT/BOMB`。推理前使用时空危险搜索屏蔽物理非法、
已知不可生存的移动，以及没有逃生路线的 BOMB。

### 3.2 D4 的基础 CNN：全图 Dueling Double DQN

棋盘被平移到以 agent 为中心的 `33×33` 画布，棋盘外编码为刚性墙。空间输入共有 11 个通道：

1. rigid walls；
2. crates；
3. visible coins；
4. self；
5. opponents；
6. bomb timer 1；
7. bomb timer 2；
8. bomb timer 3；
9. bomb timer 4；
10. active explosions；
11. predicted danger-time map。

另有 6 个标量：可放弹、回合进度、自身得分、剩余金币数、存活对手数、当前位置危险程度。

```text
11×33×33
→ Conv(11,16,3,pad=1) → ReLU
→ Conv(16,32,3,stride=2,pad=1) → ReLU
→ Conv(32,32,3,stride=2,pad=1) → ReLU
→ Flatten → Linear(2592,128) → ReLU

6 scalars → Linear(6,16) → ReLU

concat(128,16)
→ Linear(144,96) → ReLU
├→ Value:     96 → 32 → 1
└→ Advantage: 96 → 32 → 6
```

CNN 仍然只读取当前 `game_state`，没有 RNN、显式搜索树或隐藏历史状态。`n=8` 只用于训练目标，比赛时每个
时间步仍然是一次前向决策。

### 3.3 n=8 Double DQN

训练超参数：

| 项目 | 设置 |
|---|---:|
| n-step horizon | 8 |
| gamma | 0.99 |
| replay | uniform，50,000 transitions |
| batch | 256 |
| optimizer | Adam，learning rate `1e-4` |
| warmup | 8,192 transitions |
| update interval | 每 32 个环境步一次 |
| target sync | 每 8,000 个环境步 |
| loss | Smooth-L1 |
| gradient clipping | 10 |
| epsilon | 1.0 在 80,000 环境步内线性降至 0.05 |

`n=8` 来自 temporal-credit 审计：在观测到的有效攻击链中，`n=4` 无法完整覆盖 BOMB 到结果的延迟，
`n=8` 已覆盖，而 `n=16/32` 没有提供额外覆盖，反而混入更多无关 transition。因此选择 8 步，而不是继续
增大窗口。

### 3.4 D4 对称推理

Bomberman 棋盘规则对旋转和镜像近似等变，但有限训练样本下 CNN 不会自动学得完全一致。D4 推理对当前状态
生成 4 个旋转和对应镜像，共 8 个视图；网络输出映射回原动作坐标后求均值。该方法：

- 不修改权重；
- 不引入第二个模型；
- 不改变 reward 和 action mask；
- 把同一局面的几何等价判断进行平均，降低朝向偏差和偶然 Q 值尖峰。

最终确认中的平均决策耗时为 `2.71 ms`，远低于每步 `0.5 s` 的预算。

## 4. Reward 设计

CNN 路线使用稳定且可审计的真实目标奖励：

```text
COIN_COLLECTED       +1.0
KILLED_OPPONENT      +5.0
SURVIVED_ROUND       +1.0
GOT_KILLED           -5.0
KILLED_SELF          -8.0 total
CRATE_DESTROYED      +0.2
INVALID_ACTION       -0.2
BOMB_DROPPED          0.0
ordinary MOVE         0.0
```

self-kill 只记一次 `-8`，不会再叠加 `GOT_KILLED=-5`。这避免用极端击杀奖励或逐步移动惩罚支配真实目标。

中间 shaping 每步绝对值总和限制在 `0.2` 内，包括安全条件下空等、进入/逃离即时危险、接近资源前沿、
在自身安全时减少对手出口。shaping multiplier 从 Task1/2 的 `1.0`，降到 Task3 的 `0.5`，再降到 Task4
的 `0.25`，让课程后期主要由金币、击杀、死亡和生存决定。

## 5. 课程设计和 checkpoint 选择

CNN 从随机初始化开始，三个独立 replicas 使用相同超参数和预算，不同 world、agent 与 opponent seeds。

| 课程 | 环境 | 每 replica 训练局数 | 共同选择点 | pooled 课程结果 |
|---|---|---:|---:|---:|
| Task1 | coin-heaven，无对手 | 400 | round 400 | 46.34 score/round |
| Task2 | classic，无对手 | 800 | round 800 | 2.46 score/round |
| Task3-P | classic，peaceful | 600 | round 600 | 4.9567 score/round |
| Task3-C | classic，coin collector | 600 | round 150 | 3.4333 score/round |
| Task4 | classic，三个 rule-based | 1,600 | round 800 | 2.895 score/round |

每个阶段只允许从预先规定的 milestones 中选择。选择规则是：三个 replicas 的 pooled 分数距该课程最优不超过
`0.10/局`时，取最早 milestone。这样既能捕捉倒 U 型训练曲线，也避免为每个 seed 单独挑峰值。

值得写入正文的课程现象：

- Task1 三个 round-400 模型分别得到 `47.59/46.73/44.70`，说明全图 CNN 的基础资源学习稳定成立。
- Task3-P 三个模型为 `5.69/5.68/3.50`，对应击杀 `0.46/0.51/0.17` 每局，显示 seed 方差已经出现。
- Task3-C 在 round 150 达到最高 pooled 分数 `3.433`，之后 round 300/450/600 为 `3.193/3.327/3.370`；
  继续训练并不单调改善策略。
- Task4 的 round 800 优于 round 400/1200，并与 round 1600 接近，支持“中间 checkpoint 可能优于最终
  checkpoint”的选择原则。

所有课程结束后，不假定最后课程的最后一步必然最好，而是在 Task1、Task2、Task3-P、Task3-C、Task4 的
预注册课程冠军中比较最终竞赛相关表现。最终选出 `task4-r3-round0800` 作为 CNN 主干的父 checkpoint。

## 6. 从基础 CNN 到最终 D4

### 6.1 基础 CNN 的优势和问题

在一轮全新 Task4-R/M 确认中，基础 CNN 得分 `3.1575/局`，高于同批 v4 的 `2.9400/局`，并赢得
7/8 blocks。但 CNN 的自杀为 `0.2500/局`，高于 v4 的 `0.1325/局`；无效动作为 `1.2925/局`，也高于
v4 的 `0.3400/局`。因此它证明了全图 CNN 路线的总分潜力，但当时没有直接替换 v4。

### 6.2 自杀诊断

对 200 局、53 次 self-kill 的轨迹审计发现：

- 53/53 自杀前都放过自己的炸弹；
- 52/53 在放弹后的 1–4 步从“仍有静态逃生路径”变成无解；
- 35/53 包含同步移动导致的 `INVALID_ACTION`；
- 动态对手干扰证据覆盖 45/53。

问题不主要是“模型在放弹瞬间完全不知道逃生”，而是放弹后的连续移动容易被动态占位破坏。完整时域的强硬
安全约束虽然能覆盖更多自杀，却会同时压制相当一部分成功击杀；因此最终采用低介入的一步碰撞一致性条件。

### 6.3 一步碰撞 mask

在自己的炸弹仍存续时，mask 排除下一步会进入对手一步可达位置、且无法保持普通爆炸生路的移动；若交集为空，
则回退原 mask。小规模 A/B 中：

| 指标 | 原策略 | 一步碰撞 mask |
|---|---:|---:|
| score/round | 3.205 | 3.365 |
| suicides/round | 0.250 | 0.215 |
| kills/round | 0.135 | 0.145 |
| invalid/round | 1.085 | 1.030 |

该 mask 只改变 `77/65,715=0.117%` 的决策。扩大样本后，它相对原 CNN 的总分提升只有 `+0.01/局`，
但把 invalid 从 `1.265` 降到 `0.890/局`。因此报告应把它描述为精准的行为一致性清理，而不是主要得分来源。

### 6.4 末段训练：简单 control 胜过复杂对手联赛

从 `task4-r3-round0800` 继续训练时，对照臂保持三个 rule-based 对手，另一臂使用更复杂的内部策略池。
内部选择最终保留 control 的 `r2 round-150`；复杂训练分布没有产生更好的最终候选。这一结果支持一个重要判断：
对手分布更复杂不自动等于泛化更好，短剂量训练和基于官方总分的 checkpoint 选择比盲目扩大课程更可靠。

当前 D4 使用的正是这个 `control-r2-round0150` 权重。

### 6.5 D4 对称推理 A/B

首次配对 A/B 每臂 160 局：

| 指标 | 单视图 | D4 | 差值 |
|---|---:|---:|---:|
| score/round | 3.0375 | 3.3938 | +0.3563 |
| coins/round | 2.4750 | 2.9250 | +0.4500 |
| kills/round | 0.1125 | 0.0938 | -0.0188 |
| suicides/round | 0.3500 | 0.2938 | -0.0563 |
| invalid/round | 1.2063 | 0.5438 | -0.6625 |
| WAIT fraction | 4.26% | 1.93% | -2.33 pp |

D4 赢得 7/8 blocks，在 rule 与 mixed 两层、四个座位汇总上均得到支持，因此进入一次完全独立的新-seed确认。

### 6.6 D4 selection-free 最终确认

最终确认每臂 400 局，共 800 环境局；权重、mask、门槛和候选定义在结果揭晓前固定。

| 指标 | 单视图 control | D4 | 差值 |
|---|---:|---:|---:|
| score/round | 3.1750 | **3.6800** | **+0.5050** |
| coins/round | 2.4000 | **2.9925** | **+0.5925** |
| kills/round | 0.1550 | 0.1375 | -0.0175 |
| suicides/round | 0.2900 | **0.2425** | **-0.0475** |
| invalid/round | 1.1825 | **0.3800** | **-0.8025** |
| WAIT fraction | 4.47% | **1.84%** | -2.63 pp |
| mean decision time | 0.486 ms | 2.710 ms | +2.224 ms |

分层结果：

| 分层 | 单视图 | D4 | 差值 |
|---|---:|---:|---:|
| 三个 rule-based | 3.12 | **3.43** | +0.31 |
| mixed official roster | 3.23 | **3.93** | +0.70 |

D4 赢得 6/8 blocks，并通过全部预注册门：总分提升、block support、两个分层非劣、自杀不增加、invalid
不增加、击杀损失不超过 `0.02/局`、延迟低于 50 ms。正式决策为
`d4_symmetry_confirmed_as_final_internal_candidate_stop`。

### 6.7 十个新 seeds 的直接竞赛

最终 D4 又面对三个独立 seeded rule-based 对手运行 10 组、共 500 局：

| 指标 | D4 | 三个 rule-based pooled |
|---|---:|---:|
| score/round | **3.668** | 2.784 |
| score margin | **+0.884/局** | — |
| coins/round | 2.718 | 2.081 |
| kills/round | 0.190 | 0.141 |
| suicides/round | 0.278 | 0.394 |
| invalid/round | 0.476 | 6.384 |

D4 在 10/10 seeds 超过三个对手的平均分，9/10 为四名 agent 第一，10/10 进入前二。该结果不参与此前
checkpoint 选择，而是对冻结候选的绝对表现验证。

## 7. v4 的最终角色

在一组 400 局 Task4-R/M 评估中，v4 的行为画像为：

| 指标 | v4 |
|---|---:|
| score/round | 2.8825 |
| coins/round | 2.3075 |
| kills/round | 0.1150 |
| suicides/round | **0.1300** |
| invalid/round | **0.3325** |
| WAIT fraction | 26.50% |
| mean decision time | 0.306 ms |

这组数字与 D4 最终确认不是同一批 seeds，不应做精确配对差值；它们用于解释两种策略的差异。v4 的突出特点
是安全、稳定、推理便宜，但 WAIT 很高。D4 更主动地移动、争夺资源和制造局面，换来更高总分，也承受更高的
自杀风险。两者不是同一策略的强弱复制，而是两种清晰的行为类型：

- D4：最终主模型，重视总分、资源竞争和主动交互；
- v4：保留基线，重视生存稳定性和低推理成本。

报告不应把 v4 写成失败模型。它为危险预测、时空生存 mask、checkpoint 恢复和后续 CNN 设计提供了可靠基础，
也揭示了仅靠局部观察和一步 TD 容易形成保守局部最优。

## 8. 关键失败实验及其价值

正文可选 4–6 项，不必全部展开。

| 尝试 | 观察 | 最终处理 | 得到的结论 |
|---|---|---|---|
| 单纯提高 kill reward，最高到 500 | 击杀没有单调增长，部分设置反而下降 | 停止 reward 放大 | 稀有事件不能靠极端标量奖励替代可学习的 credit 与状态表示 |
| kill-chain 分层采样 | 自杀略降，但总分和击杀明显下降 | 否决 | replay 分布过度偏向稀有链会破坏常见状态学习 |
| v4 idle-WAIT 负奖励短训 | 三个 seeds 分叉明显，WAIT 仍可能上升 | 不替换 v4 | Q 排序和状态分布比一个很小的局部惩罚更重要 |
| v4 全局资源 residual | Task2 为 2.3933，几乎不高于 v4 的 2.38 | 停止 | 在冻结局部主干上外挂小 residual 不足以形成全局规划 |
| 长程强安全 mask | 能捕获更多自杀，但同时压制成功击杀 | 不启用 | 安全与进攻存在真实耦合，不能用过强规则全部切断 |
| safety replay 微调 | treatment 在 100/200 局均明显低于 uniform control | 否决 | 自杀不能仅通过增加反例采样解决 |
| 资源竞争训练 | score `+0.0625`、coins `+0.0833`，未达门槛 | 保留 control | 改变训练对手不自动产生资源泛化 |
| 降低 crate reward | 选择集曾短暂增分，但新 seeds 确认变为 `-0.13/局` | 否决 | 小样本峰值必须经 selection-free confirmation |
| productive-BOMB mask | 小样本 A/B 通过，扩大后从 3.460 降至 3.165 | 否决 | 过度过滤“暂时无目标”的炸弹会损失长期局面价值 |
| 复杂内部联赛训练 | 未超过保持三个 rule-based 的 control | 保留 control-r2 | 更复杂的训练分布并不保证更强策略 |

这些失败共同推动了最终选择：不再继续堆叠 reward 或硬编码攻击模块，而是保留简洁训练系统，把增益集中在完整
空间表示、合适的 n-step credit、严格 checkpoint 选择和低风险的对称推理上。

## 9. 可以写出的核心结论

建议最终报告围绕以下四个结论展开：

1. **完整空间表示比在局部模型上继续叠加攻击信号更有效。** 新 CNN 路线从随机初始化完成各课程，并在
   Task4 总分上稳定超过 v4 的同批基线。
2. **攻击能力不是击杀奖励的简单函数。** 极端 kill reward 和 kill-chain 采样都没有稳定奏效；`n=8`、全图
   状态和正常官方目标的组合反而产生了更完整的行为。
3. **checkpoint 选择与独立确认同训练本身一样重要。** 多个实验出现倒 U 型曲线或小样本反转；共同 milestone、
   pooled seeds 和新-seed confirmation 避免把偶然峰值当成进步。
4. **D4 对称推理是最终最可靠的单变量增益。** 它不修改权重，却在两轮独立 A/B 中同时提高总分和金币，
   降低自杀、invalid 与 WAIT，并满足延迟预算。

一句话主结论可写为：

> 我们最终得到的提升并非来自更大的击杀奖励，而是来自更完整的棋盘表示、与炸弹延迟相匹配的八步回报、
> 基于官方总分的课程 checkpoint 选择，以及对棋盘几何对称性的测试时利用。

## 10. 不得写出的结论

- 不要声称“CNN 单独带来了全部提升”；整个新路线同时改变了状态表示、网络、n-step、reward 和训练课程。
- 不要声称“击杀越多模型越强”；最终选择以官方总分为主，击杀只是其中一部分。
- 不要声称“D4 消除了自杀”；它降低了同 checkpoint 单视图策略的自杀，但仍保留主动策略的风险。
- 不要声称“课程越往后模型必然越强”；Task3-C 和 Task4 都出现非单调曲线。
- 不要把不同实验协议的分数直接做显著性差值。
- 不要编造置信区间、p-value、训练耗时或硬件利用率。没有记录的量就不写具体数字。
- 不要把 action mask 描述成独立规划器；它只排除确定非法或已知不可生存动作，Q 网络仍负责策略选择。
- 不要把 D4 写成八模型 ensemble；它是一个冻结网络在八个对称视图上的测试时平均。

## 11. 推荐的最终报告结构

最终报告必须严格采用项目说明 Section 9 规定的七个一级章节，不增设独立的 Related Work、Ablation、
Experimental Protocol 或 Discussion 一级章节。相关内容放入下面指定章节。每个章节或子章节的标题后都要标明
主要作者，例如 `Methods (Main author: Name)`；作者姓名由团队填写，不能让写作者猜测。

标题页等不计入正文。目标篇幅约为**每名成员4,000词，且不要明显超出**。标题页和正文不得使用大学 logo。
报告开头必须列出运行所需的额外 Python 库；正文必须给出完整公开代码仓库 URL。URL 尚未提供时保留明确占位符，
不得编造链接。

### 1. Introduction（负责人：待团队填写）

简要说明要解决的问题，以及它为什么有价值、为什么困难。可写 Bomberman 同时涉及稀疏奖励、炸弹延迟、空间
推理、动态多智能体交互和 0.5 秒决策限制。用一段清楚的问题陈述引出目标：在有限训练预算下得到兼顾官方总分、
生存和主动行为的学习型 agent。不要在 Introduction 提前塞入完整方法或大量结果。

### 2. Background（负责人：待团队填写）

更具体地描述任务、动作空间、官方计分和 Task1–4。概述团队考虑过的强化学习方法：tabular/linear Q-learning、
DQN、Double DQN、Dueling network、experience replay、n-step return、reward shaping、curriculum learning、
action masking、CNN 与几何对称性。说明这些方法各自适合解决什么问题，并引用相关文献。v4 和 D4 的具体实现
细节留到 Methods，不要把 Background 写成项目结果。

### 3. Project planning（负责人：待团队填写）

说明团队如何安排时间、划分子任务、协作并在截止日前整合结果。应如实写出：先建立可提交的 v4，再把后续实验
隔离到独立 checkpoint；训练结束立即生成报告；长训练与只读评估分开；使用预注册 seeds、不可覆盖输出和阶段门
控制实验范围。深度学习硬件写本地 Apple M4 与 PyTorch MPS，三个 replicas 顺序运行以控制统一内存压力，正式
比赛推理仍使用 CPU。具体成员分工和仓库协作方式没有记录，必须由团队补充，不能虚构。

### 4. Methods（负责人：待团队填写）

说明怎样把 Background 中的一般方法具体化。先介绍 v4：局部输入、MLP Dueling Double DQN、reward 和时空
安全 mask；再介绍 D4：11通道全图表示、紧凑 CNN、6个标量、n=8 Double DQN、有界 shaping、课程 checkpoint
选择、碰撞一致性 mask和八视图对称平均。解释关键设计理由并列出待比较的主要变体。实验方法也必须在本章定义：
score 为首要指标，同时记录 coins、kills、suicides、invalid、WAIT 和 latency；采用多 seeds、座位轮换、固定
milestones、pooled selection 与全新 seeds 的 selection-free confirmation。被放弃的方法及理由可以在本章末尾
简要列出，例如 v5、v4.1、v4.2a/b、v6–v10 和若干 CNN 微调路线。至少清楚呈现 v4 与 D4 两个不同 agent。

### 5. Training（负责人：待团队填写）

专门描述训练过程和加速手段。先交代 v4 的 `5,000 + 5,000 + 300` 局形成过程，再描述 CNN 三 replicas 的
Task1 400、Task2 800、Task3-P 600、Task3-C 600、Task4 1,600 局课程。写明 MPS、batch 256、uint8 replay、
每32环境步更新、target同步、epsilon schedule和课程边界清空 replay但恢复 optimizer/counters。解释 n=8、
有界辅助奖励和 milestone checkpoint 的作用。没有使用的技术不要写成已使用，例如 prioritized replay 或正式
self-play；它们只能作为考虑过但未进入最终训练的方案。

### 6. Experiments and Results（负责人：待团队填写）

这是全文最重要的章节。必须系统展示每次设计改变是提高了性能还是失败，并最终说明为什么 D4 是最佳模型。推荐
按以下顺序组织子章节，但一级标题仍只能是 `Experiments and Results`：

1. v2→v3→v4 的配对改进；
2. v4 后续候选的淘汰结果；
3. CNN Task1–4 的训练曲线和共同 milestone；
4. CNN 相对 v4 的总分提升与自杀问题；
5. self-kill 轨迹诊断和 collision mask；
6. D4 首次 A/B 与400局/臂 selection-free确认；
7. 十组新 seeds、500局直接竞赛；
8. D4 与 v4 的最终行为差异及两模型选择。

至少给出训练进展图、agent性能比较表和失败模型表。主要结果表建议保留：课程进展、D4最终A/B、十-seed直接
竞赛；v4行为画像作为补充。明确讨论遇到的困难和解决过程：奖励放大不奏效、训练曲线非单调、seed方差、课程
遗忘、小样本通过后扩大确认反转、自杀与进攻的权衡。不要只展示最好结果，也要展示关键改变为什么被否决。

### 7. Conclusion（负责人：待团队填写）

总结最可靠的发现：局部 v4 提供了安全基础；全图 CNN+n=8 扩展了资源和主动交互能力；严格 checkpoint 选择
避免倒U型训练末端；D4 对称推理在不改权重的情况下获得最终稳定提升。明确 D4 是提交主模型、v4 是保留基线。
最后给出两个具体后续方向：把对称性更早纳入训练过程；学习放弹后的动态避碰，同时保护成功进攻。再提出对下一届
游戏设置的简短建议，例如提供统一多-seed评估脚本和更明确的训练/比赛随机性控制。

## 12. 证据索引

以下文件可用于核对数值。写作者优先使用本底稿；只有需要检查字段时再读取 JSON。

```text
# 官方格式与早期训练记录
../final_project.pdf
../MODEL_A_TRAINING_GUIDE.md

# v3/v4关键原始评估（同前缀的其余seeds位于同目录）
experiments/logs/evaluations/eval-model-a-v3-balanced-classic-s4004.json
experiments/logs/evaluations/eval-model-a-v4-offense-mixed-s6001.json
experiments/logs/evaluations/eval-model-a-v4-offense-rule-s4004.json

# 课程设计与模型结构
experiments/CNN_N8_CLEAN_CURRICULUM_PLAN.md
agent_code/model_a_cnn_n8/network.py
agent_code/model_a_cnn_n8/features.py
agent_code/model_a_cnn_n8/rewards.py
agent_code/model_a_dqn/network.py
agent_code/model_a_dqn/features.py

# 课程结果
experiments/logs/course-reports/model-a-cnn-n8-clean-curriculum-s147000/task1.json
experiments/logs/course-reports/model-a-cnn-n8-clean-curriculum-s148000/task2.json
experiments/logs/course-reports/model-a-cnn-n8-clean-curriculum-s149000/task3-peaceful.json
experiments/logs/course-reports/model-a-cnn-n8-clean-curriculum-s150000/task3-coin.json
experiments/logs/course-reports/model-a-cnn-n8-clean-curriculum-s151000/task4.json

# 内部基础比较与行为诊断
experiments/logs/diagnostics/model-a-cnn-n8-final-confirmation-s152000.json
experiments/logs/diagnostics/model-a-cnn-n8-selfkill-audit-s154000/report.json
experiments/logs/diagnostics/model-a-cnn-n8-collision-mask-ab-s159000/report.json
experiments/logs/diagnostics/model-a-cnn-n8-collision-mask-final-s160000/report.json

# D4 独立确认
experiments/logs/diagnostics/model-a-cnn-n8-d4-symmetry-ab-s170000/report.json
experiments/logs/diagnostics/model-a-cnn-n8-d4-symmetry-final-s171000/report.json
experiments/logs/diagnostics/model-a-cnn-n8-d4-rule-10seed-s172000/report.json
```

## 13. 最终写作检查清单

交稿前逐项检查：

- [ ] D4 明确写成最终主模型，v4 明确写成保留基线。
- [ ] D4 的 checkpoint、CNN、mask、八视图平均关系写对。
- [ ] 没有把 n=8 误写成八帧输入或八步在线规划。
- [ ] 没有把 D4 误写成八个网络。
- [ ] D4 最终确认使用 400 局/臂，结果为 3.680 对 3.175。
- [ ] 十-seed评估使用 500 局，D4 平均 3.668，9/10 第一。
- [ ] v4 被写成有价值的安全基线，而不是失败品。
- [ ] 失败实验服务于设计结论，没有写成冗长道歉或实验流水账。
- [ ] 不包含本项目两条自研路线和官方内置对手之外的模型叙述。
- [ ] 语言自然，少套话，不提写作工具，不虚构统计量。
