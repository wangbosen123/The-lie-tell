# Code Review 中文說明：Lie Diffuser Actor

**論文：** *The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space*  
**GitHub：** <https://github.com/tars3017/lie-diffuser-actor>  
**本地 repo：** `C:\Users\user\Documents\Codex\2026-09-12\the-lie-we-tell-correcting-the\work\lie-diffuser-actor`  
**Review 範圍：**  Code review，聚焦資料處理、模型架構、training、inference、tangent-space exponential map、實驗比較 code。

## 整體結構

這個 repo 主要分成兩條線：

| 區塊 | 路徑 | 用途 |
|---|---|---|
| CALVIN 主實驗 | `calvin/` | 論文主要 Lie Diffuser Actor 實作，包含資料、模型、訓練、評估與 ablation |
| OpenVLA-OFT / LIBERO | `openvla_oft/` | 用 OpenVLA-OFT 做 baseline、Euclidean score matching、Lie score matching 的比較 |
| 實驗設定 | `calvin/configs/`, `openvla_oft/lda_oft/configs/` | 控制 full method 與 ablation |
| task metadata | `tasks/` | CALVIN task bounds 等輔助檔 |

CALVIN 是比較完整的主方法 implementation；OpenVLA-OFT 則把 tangent-space score matching 做成比較乾淨的 action head，適合看 contribution 的數學實作。

---

## 1. 模型架構寫在哪裡？

### 對應 code 位置

CALVIN 主模型：

| 模組 | 檔案 / 位置 |
|---|---|
| top-level policy | `calvin/lda/model/diffuser_actor.py:38` 的 `DiffuserActor` |
| visual / language / gripper encoder 呼叫 | `calvin/lda/model/diffuser_actor.py:135` 的 `encode_inputs()` |
| diffusion prediction head | `calvin/lda/model/diffuser_actor.py:688` 的 `DiffusionHead` |
| model forward | `calvin/lda/model/diffuser_actor.py:440` 的 `forward()` |
| GAT planner 條件分支 |`calvin/lda/model/diffuser_actor.py:114`, `:469` |(額外看robot arm joints/link geometry，建立 condition)
| GAT encoder | `calvin/lda/gat/models/gat_encoder.py` |
| visual encoder 相關 | `calvin/lda/encoder/` |

OpenVLA-OFT action head：

| 模組 | 檔案 / 位置 |
|---|---|
| L1 baseline head | `openvla_oft/prismatic/models/action_heads.py:367` 的 `L1RegressionActionHead` |
| DDIM diffusion head | `openvla_oft/prismatic/models/action_heads.py:427` 的 `DiffusionActionHead` |
| SE(3) score matching head | `openvla_oft/prismatic/models/action_heads.py:537` 的 `SE3ScoreMatchingActionHead` |

### 做法流程

CALVIN 的模型大致是：

RGB(相機影像這邊論文沒有提到) + point cloud(3D資訊點雲圖) + instruction(prompt) + 
gripper history(end-effector position + rotation + gripper state) + joints(整隻手臂每個關節/link在哪裡、朝哪裡(condition 的概念))
    -> Encoder 抽 visual/context features
    -> optional GraphPlanner + GATEncoder 建 local graph condition
    -> noisy trajectory / pose query
    -> DiffusionHead cross-attention / self-attention
    -> predict position score, rotation score, gripper openness


`DiffuserActor` 裡有兩個重要 switch：
```text
use_gat = 1 / 0
diffusion_space = lie / euclidean
```
`use_gat=1` 時會使用 graph planner 和 GAT encoder，讓 trajectory prediction 多一個 robot-scene graph condition。
`diffusion_space="lie"` 時，trajectory 在模型內被視為 SE(3) tangent-space 6 維向量：

```text
[translation tangent 3D, rotation tangent 3D]
```

`diffusion_space="euclidean"` 時，則使用一般 Euclidean DDPM，rotation 使用 6D rotation representation。

OpenVLA-OFT 的模型架構比較簡單。它不改整個 OpenVLA backbone，而是在 action token hidden states 後面接 action head：

```text
image + language prompt + action placeholder tokens
    -> OpenVLA backbone
    -> action token hidden states
    -> action head
    -> action / score prediction
```

---

## 3. Training 寫在哪裡？

### 對應 code 位置

CALVIN training：

| 功能 | 檔案 / 位置 |
|---|---|
| shell entrypoint | `calvin/scripts/train.sh` |
| CALVIN args / dataset / optimizer | `calvin/train.py` |
| 建 train / val dataset | `calvin/train.py:40` 的 `get_datasets()` |
| collate function | `calvin/train.py:224` 的 `traj_collate_fn()` |
| 通用 training loop | `calvin/lda/engine.py:104` 的 `BaseTrainTester.main()` |
| 建模型 | `calvin/lda/trainer.py:175` 的 `get_model()` |
| 單步 training | `calvin/lda/trainer.py:202` 的 `train_one_step()` |
| Lie loss | `calvin/lda/model/diffuser_actor.py:528` 的 `_compute_loss_lie()` |
| Euclidean ablation loss | `calvin/lda/model/diffuser_actor.py:630` 的 `_compute_loss_euclidean()` |

OpenVLA-OFT training：

| 功能 | 檔案 / 位置 |
|---|---|
| shell entrypoint | `openvla_oft/scripts/train.sh` |
| YAML wrapper | `openvla_oft/lda_oft/train.py` |
| config 轉 train args | `openvla_oft/lda_oft/train.py:20` 的 `build_argv()` |
| 主 finetuning script | `openvla_oft/vla-scripts/finetune.py` |
| training forward / loss | `openvla_oft/vla-scripts/finetune.py:274` 的 `run_forward_pass()` |
| SE(3) score matching loss | `openvla_oft/vla-scripts/finetune.py:440` |
| SE3 head 初始化 | `openvla_oft/vla-scripts/finetune.py:1128` |

### 做法流程

CALVIN training 流程：

```text
scripts/train.sh
    -> 讀 YAML config
    -> torchrun train.py
    -> train.py 建 CalvinDataset
    -> engine.py 建 DataLoader / model / optimizer / DDP
    -> trainer.py 每一步呼叫 model
    -> DiffuserActor.forward()
    -> 根據 diffusion_space 選 Lie loss 或 Euclidean loss
    -> backward / optimizer step / checkpoint
```

`train_one_step()` 裡會先處理 trajectory：

- 若不是 keypose-only，就把第一個 trajectory step 拿掉。
- 取 current gripper 或 gripper history。
- 呼叫 model。
- model 回傳的第一個值就是 loss tensor。
- 直接 `out.backward()`。

Lie training 的核心流程：

```text
GT trajectory quaternion pose
    -> SE(3)
    -> tangent vector
    -> tangent-space normalization
    -> sample tangent noise z
    -> rt = r0 @ Exp(sqrt_alpha * z)
    -> model predict score / movement
    -> target = -z
    -> weighted MSE + gripper BCE
```

OpenVLA-OFT training 流程：

```text
scripts/train.sh
    -> lda_oft/train.py 讀 YAML
    -> vla-scripts/finetune.py
    -> RLDS dataloader 給 batch
    -> sample_noisy_actions()
    -> OpenVLA forward 取得 action token hidden states
    -> SE3ScoreMatchingActionHead.predict_score()
    -> loss 對齊 -noise
    -> LoRA / action head optimization
```

SE(3) score matching loss：

```text
target_se3 = -noise_se3
loss = 20 * MSE(translation score)
     + 10 * MSE(rotation score)
     + BCE(gripper logits, gripper target)
```

---

## 4. Inference 寫在哪裡？

### 對應 code 位置

CALVIN inference：

| 功能 | 檔案 / 位置 |
|---|---|
| eval shell entrypoint | `calvin/scripts/eval.sh` |
| CALVIN benchmark runner | `calvin/lda/eval/evaluate_policy.py` |
| 評估整個 policy | `calvin/lda/eval/evaluate_policy.py:107` 的 `evaluate_policy()` |
| 評估一條 instruction sequence | `calvin/lda/eval/evaluate_policy.py:176` 的 `evaluate_sequence()` |
| 每個 subtask rollout | `calvin/lda/eval/evaluate_policy.py:215` 的 `rollout()` |
| policy wrapper | `calvin/lda/eval/evaluate_model.py` |
| 單次模型 step | `calvin/lda/eval/evaluate_model.py:146` 的 `step()` |
| 產生 trajectory | `calvin/lda/model/diffuser_actor.py:314` 的 `compute_trajectory()` |
| Lie sampling | `calvin/lda/model/diffuser_actor.py:209` 的 `_conditional_sample_lie()` |
| Euclidean sampling | `calvin/lda/model/diffuser_actor.py:266` 的 `_conditional_sample_euclidean()` |

OpenVLA-OFT / LIBERO inference：

| 功能 | 檔案 / 位置 |
|---|---|
| eval shell entrypoint | `openvla_oft/scripts/eval.sh` |
| YAML eval wrapper | `openvla_oft/lda_oft/eval.py` |
| LIBERO eval runner | `openvla_oft/experiments/robot/libero/run_libero_eval.py` |
| 載模型與 action head | `openvla_oft/experiments/robot/libero/run_libero_eval.py:145` 的 `initialize_model()` |
| 跑 episode | `openvla_oft/experiments/robot/libero/run_libero_eval.py:290` 的 `run_episode()` |
| 跑 task success rate | `openvla_oft/experiments/robot/libero/run_libero_eval.py:367` 的 `run_task()` |
| 建 action head | `openvla_oft/experiments/robot/openvla_utils.py:485` 的 `get_action_head()` |
| query VLA action | `openvla_oft/experiments/robot/openvla_utils.py:745` 的 `get_vla_action()` |
| HF model predict action | `openvla_oft/prismatic/extern/hf/modeling_prismatic.py:1081` 的 `predict_action()` |
| SE(3) score matching inference | `openvla_oft/prismatic/extern/hf/modeling_prismatic.py:880` 的 `_run_se3_score_matching_prediction()` |

### 做法流程

CALVIN inference 流程：

```text
scripts/eval.sh
    -> evaluate_policy.py
    -> rollout()
    -> prepare RGB / point cloud / proprio / language embedding / joints
    -> DiffusionModel.step()
    -> DiffuserActor.forward(run_inference=True)
    -> compute_trajectory()
    -> conditional_sample()
    -> env.step(action)
    -> task oracle 判斷成功與否
```

在 `DiffusionModel.step()` 中，模型會先建立假的空 trajectory mask，因為 inference 時沒有 GT trajectory；接著將 observation 整理成模型需要的格式，呼叫 policy 的 `run_inference=True` 分支。

OpenVLA-OFT / LIBERO inference 流程：

```text
scripts/eval.sh
    -> lda_oft/eval.py
    -> run_libero_eval.py
    -> initialize_model()
    -> run_episode()
    -> get_vla_action()
    -> model.predict_action()
    -> 若 action_head 是 SE3ScoreMatchingActionHead
        -> _run_se3_score_matching_prediction()
    -> action unnormalize
    -> env.step(action)
```

SE(3) inference 使用 annealed Langevin dynamics：

```text
sample initial tangent noise
    -> Exp map 到 SE(3) matrix
    -> 每個 timestep:
        -> Log map 回 tangent 給模型
        -> predict score
        -> update = step_size * score / sigma + noise
        -> Exp(update)
        -> pose = pose @ Exp(update)
    -> final pose Log map 回 tangent action
    -> gripper sigmoid
```

---

## 5. 貢獻：tangent-space prediction head 的 exponential map 怎麼做？

### 對應 code 位置

CALVIN 主方法：

| 功能 | 檔案 / 位置 |
|---|---|
| Lie training loss | `calvin/lda/model/diffuser_actor.py:528` 的 `_compute_loss_lie()` |
| forward diffusion 用 Exp map | `calvin/lda/model/diffuser_actor.py:570` |
| Lie metrics wrapper | `calvin/lda/diffusion/lie/metrics/se3.py` |
| Lie group composition | `calvin/lda/diffusion/lie/utils/ops.py` |
| NormalSE3 noise | `calvin/lda/diffusion/lie/dist/se3.py` |

OpenVLA-OFT 清楚版 exponential map：

| 功能 | 檔案 / 位置 |
|---|---|
| SO(3) exponential map | `openvla_oft/prismatic/models/action_heads.py:16` 的 `so3_exp_map()` |
| SO(3) log map | `openvla_oft/prismatic/models/action_heads.py:58` 的 `so3_log_map()` |
| SE(3) exponential map | `openvla_oft/prismatic/models/action_heads.py:161` 的 `se3_exp_map()` |
| SE(3) log map | `openvla_oft/prismatic/models/action_heads.py:213` 的 `se3_log_map()` |
| SE(3) forward noise | `openvla_oft/prismatic/models/action_heads.py:258` 的 `se3_add_noise()` |
| SE(3) update step | `openvla_oft/prismatic/models/action_heads.py:272` 的 `se3_compose_step()` |
| score matching head | `openvla_oft/prismatic/models/action_heads.py:537` 的 `SE3ScoreMatchingActionHead` |

### 做法流程

CALVIN 的 tangent-space score matching：

```text
GT pose: [x, y, z, quaternion]
    -> lie_metrics.as_lie()
    -> SE(3) group element
    -> lie_metrics.as_repr(..., "tan")
    -> 6D tangent vector
    -> normalize tangent vector
    -> sample z from NormalSE3
    -> rt = r0 @ Exp(sqrt_alpha * z)
    -> model predict 6D tangent score
    -> target = -z
```

核心 forward diffusion code：

```python
rt_flat = ops.add(r0_flat, SE3.exp_map(sqrt_alphas_t * zt_flat))
```

位置：`calvin/lda/model/diffuser_actor.py:570`

意思是：

```text
noisy pose = clean pose composed with Exp(noise in tangent space)
```

OpenVLA-OFT 的 `se3_exp_map()` 寫得比較直觀。輸入是：

```text
xi = [v_x, v_y, v_z, omega_x, omega_y, omega_z]
```

其中：

- `v` 是 translation tangent。
- `omega` 是 rotation tangent。

`se3_exp_map()` 做的事：

1. 用 `omega` 建 skew-symmetric matrix `K`。
2. 算 `theta = ||omega||`。
3. 用 Rodrigues formula 算 rotation matrix：

```text
R = I + sin(theta)/theta * K + (1 - cos(theta))/theta^2 * K^2
```

4. 算 SE(3) left Jacobian：

```text
V = I + (1 - cos(theta))/theta^2 * K + (theta - sin(theta))/theta^3 * K^2
```

5. translation 是：

```text
t = V @ v
```

所以 SE(3) exponential map 的結果是：

```text
xi in tangent space -> (R, t) in SE(3)
```

在 forward diffusion 中：

```python
R0, t0 = se3_exp_map(xi0)
R_noise, t_noise = se3_exp_map(sqrt_alpha_t * z)
Rt, tt = se3_compose(R0, t0, R_noise, t_noise)
return se3_log_map(Rt, tt)
```

位置：`openvla_oft/prismatic/models/action_heads.py:258`

也就是：

```text
xi_t = Log(Exp(xi_0) @ Exp(sqrt_alpha * z))
```

Euclidean ablation 則不做 Exp/Log：

```python
noisy_xi = xi0 + sqrt_alpha * z
```

這個差異正是論文要比較的重點：action pose 不應該直接當 Euclidean vector 處理，而應該尊重 SE(3) manifold geometry。

---

## 6. 實驗比較 code

### 對應 code 位置

CALVIN 實驗比較：

| 實驗 | config |
|---|---|
| full method, ABC->D | `calvin/configs/lda_abc_d.yaml` |
| full method, ABCD->D | `calvin/configs/lda_abcd_d.yaml` |
| w/o Lie, ABC->D | `calvin/configs/ablation_no_lie_abc_d.yaml` |
| w/o Lie, ABCD->D | `calvin/configs/ablation_no_lie_abcd_d.yaml` |
| w/o GAT, ABC->D | `calvin/configs/ablation_no_gat_abc_d.yaml` |
| w/o GAT, ABCD->D | `calvin/configs/ablation_no_gat_abcd_d.yaml` |

OpenVLA-OFT / LIBERO 實驗比較：

| 實驗 | config |
|---|---|
| OpenVLA-OFT baseline | `openvla_oft/lda_oft/configs/oft_baseline_libero10.yaml` |
| Euclidean score matching | `openvla_oft/lda_oft/configs/oft_euclidean_sm_libero10.yaml` |
| Lie score matching | `openvla_oft/lda_oft/configs/oft_lie_sm_libero10.yaml` |

YAML wrapper：

| 功能 | 檔案 / 位置 |
|---|---|
| config loader | `openvla_oft/lda_oft/config.py` |
| train wrapper | `openvla_oft/lda_oft/train.py` |
| eval wrapper | `openvla_oft/lda_oft/eval.py` |

### 做法流程

CALVIN 比較方式主要靠 config flags：

```yaml
use_gat: 1
diffusion_space: lie
loss_formulation: new
```

full method：

```text
use_gat = 1
diffusion_space = lie
loss_formulation = new
```

w/o Lie：

```text
use_gat = 1
diffusion_space = euclidean
```

w/o GAT：

```text
use_gat = 0
diffusion_space = lie
```

這些 flags 會在 `DiffuserActor` 裡控制模型 branch：

- `use_gat` 控制是否啟用 GraphPlanner / GATEncoder。
- `diffusion_space` 控制走 Lie score matching 還是 Euclidean DDPM。
- `loss_formulation` 控制 corrected loss 或舊版 loss。

OpenVLA-OFT 的比較則更乾淨，主要靠三個 YAML：

baseline：

```yaml
use_l1_regression: true
use_diffusion: false
use_se3_score_matching: false
```

Euclidean score matching：

```yaml
use_l1_regression: false
use_diffusion: false
use_se3_score_matching: true
score_matching_lie_group: false
```

Lie score matching：

```yaml
use_l1_regression: false
use_diffusion: false
use_se3_score_matching: true
score_matching_lie_group: true
```

也就是同一個 `SE3ScoreMatchingActionHead`，只用 `score_matching_lie_group` 切換數學路徑。

---

## Code Review 流程順序規劃

1. `README.md`
2. `calvin/README.md`
3. `calvin/scripts/package_calvin.py`
4. `calvin/lda/data/calvin.py`
5. `calvin/lda/model/diffuser_actor.py` (Important => paper main method)
6. `calvin/lda/trainer.py`
7. `calvin/lda/eval/evaluate_model.py`
8. `openvla_oft/prismatic/models/action_heads.py` (Important => exponential map / Lie vs Euclidean) 
9. `openvla_oft/vla-scripts/finetune.py`
10. `openvla_oft/prismatic/extern/hf/modeling_prismatic.py`
