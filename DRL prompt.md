# ✅ 任务：在现有 notebook 后追加 “DRL-based cost minimization (paper 5.3 / 3.3.3)” 代码（不破坏上面任何 cell）

你正在修改 Jupyter notebook：`半监督学习.ipynb`

## 0) 总目标（Business Goal）
我已经完成了前面 Cox/Weibull + hazard risk 构造、以及 tabular Q-learning（基于离散 state + P0/P1/P2）的部分。
现在需要你在 notebook 的 DRL 章节后面**新增**一整套 “DRL-based cost minimization” 代码，实现连续 state 的 DRL 训练，并补齐我缺失的：
- ✅ “minor repair” 的连续定义（age rollback + hazard reduction + diminishing returns）
- ✅ 2 个 sanity check（验证小修不会过强 + 小修边际收益递减）
要求：**不要修改/重排/破坏上面所有已有 cell**，只允许在后面 append 新 cell。

> Paper check（你必须遵守的论文约束）：
> - DRL 使用连续 state（hazard risk + age），用 MLP 近似 Q
> - MLP: one hidden layer with 512 nodes, `sigmoid` activation
> - output layer activation: `softmax`
> - train for 200 epochs
> - if `action == 2 (major repair)` then terminate the epoch/episode immediately
> - DRL convergence depends on “minor repair definition”
> （这些来自论文 DRL 小节描述） :contentReference[oaicite:1]{index=1}

---

## 1) Hard Constraints（必须严格执行）
1. **Do NOT edit earlier cells**：包括 data loading、scaling、Weibull/Cox、hazard risk、KMeans、P0/P1/P2、tabular Q-learning 等。  
2. 所有新逻辑必须写在 NEW cells（append at the end / after DRL markdown）。  
3. 不允许新增外部文件依赖（除非保存一个新的 notebook 输出文件）。  
4. 设置 `random seed`（`random`, `numpy`, `torch`）保证 reproducibility。  
5. 输出必须可运行（假设上面 cell 已经 run 过并产生 hazard risk dataframe）。

---

## 2) 先做 Notebook Scan（你需要自动定位变量）
你要先扫描 notebook，定位这些变量/列名（不要让我手动改一堆变量名）：
- `df` / `df_process` / `df_final`（whatever exists）中：
  - `unit id`（例如 `unit_nr`）
  - `age`（例如 `time_cycles`）
  - `hazard risk` 的连续列（例如 `hazard_risk` / `HR` / `logHR_scaled` / 你已有的 risk 输出列）
- 如果找不到统一的 risk 列名：
  - 就在新 cell 里用一个变量 `RISK_COL` 指向你找到的那列，并打印确认（`print(RISK_COL, AGE_COL, UNIT_COL)`）。

---

## 3) Reward / Cost（必须跟 notebook Eq.(9) 一致，禁止自作主张）
⚠️ 重要：paper 的 DRL 小节只说 reward 与上一 case study “exactly the same”，并没有重复数值。为避免版本混淆，**以 notebook 为准**。

你必须在新 DRL cell 里：
- 如果 notebook 已存在 `R_action`（numpy array），就直接复用：
  - `R_action = existing_R_action`
- 若不存在才 fallback 到：
  - `R_action = np.array([0.0, -20.0, -40.0])`
并在输出里 `print("Using R_action:", R_action)`。

Action meaning 固定：
- `a=0`: `no repair`
- `a=1`: `minor repair`
- `a=2`: `major repair`

Reward:
- `r = R_action[a]`（直接用该数组，不要写 -100）

---

## 4) DRL Environment（连续 state 的 step 逻辑）
你要实现一个轻量 `Env`（可以是 class `MaintenanceEnv`）输出 transition：
`(state, action, reward, next_state, done, info)`

### 4.1 State definition（paper-aligned）
- `state s = (h, age)`  where:
  - `h = hazard risk` (continuous)
  - `age = time_cycles` (or equivalent)
- 你可以选择把 state 组装为 numpy array: `np.array([h, age], dtype=np.float32)`

### 4.2 Episode sampling（offline trajectory driven）
每个 episode 从某个 `unit` 开始：
- 随机选一个 `unit_id`
- 从该 unit 的某个起始 index 开始（默认从最早 cycle 开始也可以）
- episode 内以 “dataset trajectory” 驱动下一个状态

### 4.3 Transition rules（核心）
**Case A: `action == 0` (no repair)**
- 按该 unit 的真实轨迹走到下一行：`idx += 1`
- `next_state = (h_next, age_next)` 来自下一行
- 如果已经是该 unit 最后一行：`done=True`

**Case B: `action == 1` (minor repair) — 必须实现：imperfect repair + diminishing returns**
你需要新增一个独立的 minor repair 机制（不要用旧的 `P1` matrix）：

Maintain per-episode counter:
- `k_minor` = number of minor repairs taken so far in this episode

Diminishing returns parameters（默认值写成常量，方便我以后改）：
- `r0 = 0.30`
- `eta0 = 0.25`
- `decay = 0.15`

Compute effectiveness:
- `r_k = r0 * exp(-decay * k_minor)`  (age rollback ratio)
- `eta_k = eta0 * exp(-decay * k_minor)` (hazard reduction ratio)

Transform:
- `age_after = floor((1 - r_k) * age_current)`
- `h_after = max(h_min, (1 - eta_k) * h_current)`

`h_min` 定义（防止“better than new”）：
- 通过数据自适应计算，例如：
  - `h_min = percentile(hazard_risk among early cycles, e.g. first 5 cycles, q=1%)`
  - or robust minimum clip `max(global_min, 1e-6)`
并打印 `h_min`。

**Mapping back to trajectory（推荐做法，保证 realism）**
Minor repair 后不要凭空产生一个 state，你要把 `(age_after)` 映射回同一 unit 的历史轨迹：
- 在该 unit 的 dataframe 中，找 `time_cycles` 最接近 `age_after` 的 row（prefer `<= age_current`）
- 得到 `h_from_traj`
- 最终 `h_next = max(h_min, (1 - eta_k) * h_from_traj)`
- `age_next = matched_time_cycles`
- episode 继续（`done=False`，除非 edge case）

Fallback（如果 mapping 找不到）：
- `next_state = (h_after, age_after)` 但要在 `info` 标注 `fallback=True`

**Case C: `action == 2` (major repair)**
- `reward = R_action[2]`（-40）
- `done=True` immediately（paper requirement）
- `next_state` 可以返回 current_state 或 None，但必须终止 episode

---

## 5) 两个 Sanity Checks（必须新增 NEW cells 并打印清晰诊断）
### Sanity Check #1: minor repair should reduce hazard but not below `h_min`
- Sample `N=1000` random states from dataset
- Apply one minor repair with `k_minor=0` (do NOT advance trajectory)
- Validate:
  - `h_after <= h_before + tol`
  - `h_after >= h_min`
- Print:
  - `num_violations_reduce`
  - `num_violations_lower_bound`
  - show top 5 violating examples (unit_id, age, h_before, h_after)
- If violations exist: print suggestions:
  - decrease `eta0` or increase `h_min` quantile

### Sanity Check #2: diminishing returns (repeated minor repairs yield decreasing benefit)
- Pick `M=200` random starting states
- Apply minor repair repeatedly `K=5` times (do NOT advance along trajectory), updating `k_minor`
- Compute deltas: `delta_i = h_{i-1} - h_i`
- Validate mean deltas are non-increasing:
  - `mean(delta_1) >= mean(delta_2) >= ...` within tolerance
- Print a small table:
  - step, mean_delta, std_delta, violation_rate
- If too many violations: suggest increasing `decay`

---

## 6) DRL Model（paper strict: softmax output）
Framework: Prefer `PyTorch`.

Network `QNetwork`:
- input dim = 2
- hidden layer = 512, activation = `torch.sigmoid`
- output dim = 3, activation = `softmax(dim=-1)`

⚠️ Note（写在代码注释里）：standard DQN uses linear outputs, but paper uses `softmax`, so here we treat softmax outputs as bounded Q-like scores for TD learning (paper-faithful).

Replay:
- `ReplayBuffer(capacity=100000)`
- store `(state, action, reward, next_state, done)`

Policy:
- `epsilon-greedy`: with prob `epsilon` random action, else `argmax(Q(state))`
- epsilon schedule: reuse notebook’s RL section if present; else:
  - `eps_max=1.0`, `eps_min=1/200`, `eps_decay=0.1` (exponential decay)

Training:
- `epochs = 200` (paper)
- `gamma = 0.99`
- optimizer: `torch.optim.SGD` (paper mentions SGD), default `lr=1e-3`
- batch size: `64`
- loss:
  - `q_sa = Q(state)[action]`
  - `target = reward + gamma * (1 - done) * max(Q(next_state))`
  - `loss = (target - q_sa)^2` mean over batch
- For stability: clip gradients (e.g. `torch.nn.utils.clip_grad_norm_`)

Episode termination:
- If `action == 2`: terminate immediately (paper)
- Also terminate if unit reaches end-of-trajectory
- Add `max_steps_per_episode=400` safety cap

---

## 7) Evaluation（必须给我可读的结果）
Split by `unit_id`:
- train/test = 80/20 (fixed seed)

Report:
1) `avg_total_cost_per_episode` on test units under greedy policy (`epsilon=0`)
2) action frequency: `%a0, %a1, %a2`
3) action preference by hazard bins:
   - bin hazard into low/med/high and print most common action per bin

Plot (matplotlib):
- training `loss` curve
- `avg_episode_cost` per epoch

---

## 8) Deliverable（输出文件）
为了不破坏原文件：
- Save a new notebook: `半监督学习_DRL.ipynb`

---

## 9) Acceptance Criteria（验收标准）
- 上面所有旧 cell 不变（no diffs in earlier sections）
- DRL 部分新增：`MaintenanceEnv`, `ReplayBuffer`, `QNetwork`, training loop, evaluation
- minor repair = age rollback + hazard reduction + diminishing returns（必须有 `k_minor`）
- 两个 sanity check 都有，并打印 violations + 调参建议
- Reward 来自 notebook `R_action`（默认 `[0,-20,-40]`），禁止写 `-100`
- Network 输出必须是 `softmax`
- epochs=200，major repair action 直接终止 episode（paper）

开始执行时，先 scan notebook 找到 `UNIT_COL`, `AGE_COL`, `RISK_COL`, `R_action`，然后 append 新 cells 实现以上全部内容。
