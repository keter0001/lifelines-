你是一个资深深度强化学习工程师。我要你对我当前的 DRL/DQN notebook 进行“静态改造”（只改代码，不运行、不训练、不做任何需要大量算力的执行）。请你严格遵守：**不要运行任何代码**，只输出修改后的代码片段/替换补丁，以及你修改每一处的理由（用简洁要点）。

【背景与问题】
我现在的 DRL 代码出现典型的“动作塌缩/策略退化成单一动作（经常 a=0）”问题。我的实现存在以下结构性风险：
1) 我把 DQN 的 Q-network 输出层做成 softmax（输出像概率，限制在(0,1)且和为1），但又用 TD 回归损失（Bellman target）去训练。这是 DQN 里常见错误，会导致目标与输出空间不匹配、梯度饱和、动作塌缩。
2) 为了配合 softmax 输出，我把 reward 映射到 [0.1,0.9] 并对 TD target 做 sigmoid 压缩，还缩放 gamma（gamma_scaled）。这会进一步把价值差异“挤扁”，导致学习信号变弱、policy 更容易塌缩。
3) 我训练时 reward 用了 action_cost + state_cost，但日志统计 cost 只累计 action_cost，导致评估/对照论文时出现误判。
4) 我从数据估计的 P0 可能使 failure 在 no-action 下接近吸收态；如果环境未按论文口径处理终止/重置与 failure penalty，会加剧偏差。但本次先以修正 DQN 结构为主。

【你的任务目标】
把我的 notebook 代码改成“标准、稳健”的 DQN 实现：
- Q-network 输出：**线性输出 3 个 Q 值（不使用 softmax）**
- TD target：**标准形式** target = r + gamma*(1-done)*max_a' Q_target(next_state,a')
- Loss：用 MSE 或 Huber（建议 Huber）
- Exploration：epsilon-greedy 保留（或可选 boltzmann exploration，但 softmax 只能用于 action sampling，而不是网络输出层）
- 保留：replay buffer、target network、gradient clipping（若已有）
- 评估/日志：cost 必须与训练 reward 同口径（把 state_cost 也计入 cost）
- 不要运行：只给出改好的代码，确保能在我的 notebook 中直接替换。

【输入给你的资料】
我的 notebook 文件：/mnt/data/colab DRL学习.ipynb
请你读取该 ipynb，定位与 DRL 相关的 cell，尤其是：
- Q-network 定义（含 softmax）
- TD target / loss 计算（含 reward 映射、sigmoid 压缩、gamma_scaled）
- 训练循环（epsilon-greedy、replay sampling、target update）
- 日志统计（episode_cost / total_cost 等）

【输出格式要求（非常重要）】
1) 先给一个“修改清单”（bullet list），说明你要改哪些地方（按文件/Cell 顺序或按函数名）。
2) 然后给出“可直接替换的代码块”，每个代码块标明：替换哪个函数/类/段落（原位置描述）。
3) 每个改动后面用 1-2 句说明“为什么这样改”（例如：softmax 不适用于 Q 回归等）。
4) 不要训练，不要调用 fit，不要跑任何循环；如需要验证，用注释写“如何验证”，但不要实际执行。

【必须落地的具体改动点】
A) QNetwork / model 输出层修改
- 删除输出层 softmax
- 最后一层改成 Linear(out_features=3)
- forward 返回 raw Q-values

B) TD target 和 loss 修改
- 删除 reward 映射到 [0.1,0.9] 的逻辑
- 删除 sigmoid(target_raw) 或 target 压缩逻辑
- 删除 gamma_scaled，使用原 gamma
- 用标准 DQN target： r + gamma*(1-done)*max(Q_target(next_state))
- 建议 loss：HuberLoss（SmoothL1Loss）

C) Action selection（决策仍然是维修决策）
- epsilon-greedy：greedy action = argmax(Q(s,:))
- 如需要 tie-break：argmax 平局随机（可选）

D) 日志 cost 口径修正
- 训练里如果 reward = action_reward + state_cost（通常为负）
- 那么 cost 应该累计 -(action_reward + state_cost)，而不是只累计 -action_reward
- 额外输出：minor/major/no-action 计数与比例（用于对照论文表格）

E) done/终止机制（保持你已有逻辑，但让 target 处理 done）
- target 里乘 (1-done)，done=True 时 next_q=0
- 如果你环境规定 major repair done=True，则在 major repair 步结束 episode

【请你额外注意的常见坑（也要在修改后检查）】
- Q-network 与 target network 在评估时要用 .eval()（如果是 PyTorch）
- 采样 batch 时要确保 done 是 float(0/1)
- 状态输入形状：batch x state_dim
- 如果状态是 [hazard_risk, age]（连续），要保证归一化/标准化（可选：只加轻量归一化，不训练时不强制）

【最终交付物】
- 你返回“修改后的关键代码段”，让我直接复制粘贴回 notebook。
- 你不需要生成完整 ipynb 文件，但要保证我照着替换能跑。
- 不要运行。

开始执行：请先打开 /mnt/data/colab DRL学习.ipynb，定位相关代码并按上述要求输出补丁。
