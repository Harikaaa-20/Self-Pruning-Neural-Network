# The Self-Pruning Neural Network

**Tredence Analytics — AI Engineering Case Study**

---

## 1. Overview

This project implements a **self-pruning neural network** that dynamically removes unnecessary connections during training. Unlike traditional pruning (performed after training), this model learns a sparse structure *end-to-end* using a learnable gating mechanism.

The approach is evaluated on the CIFAR-10 dataset, demonstrating a **controlled trade-off between model sparsity and accuracy**.

---

## 2. Methodology

### PrunableLinear Layer

A custom linear layer is implemented where each weight is associated with a learnable parameter `gate_score`.

During the forward pass:

```python
gates = torch.sigmoid(gate_scores)
pruned_weights = weight * gates
```

* Each gate ∈ (0, 1) controls whether a connection is active
* Gates approaching 0 effectively remove the corresponding weight
* Both weights and gates are trained jointly via backpropagation

---

### Sparsity-Inducing Loss

To encourage pruning, an L1 penalty is applied on the gate values:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where:

* `SparsityLoss = sum of all gate values`
* λ controls the strength of pruning

#### Why L1 encourages sparsity

L1 regularization applies a constant penalty on all gates, forcing each connection to justify its contribution. Connections that do not significantly reduce classification loss are driven toward zero, resulting in a sparse network.

---

## 3. Training and Stabilization

Initial experiments showed unstable behavior, where sparsity rapidly collapsed to ~99% even at small λ values. This was caused by aggressive optimization dynamics on gate parameters.

To address this:

* Smaller λ values were explored
* Training dynamics were stabilized
* Pruning behavior became gradual instead of collapsing

This resulted in a **smooth and controllable sparsity–accuracy trade-off**.

---

## 4. Results

| λ (Lambda) | Test Accuracy | Sparsity (< 1e-2) |
| ---------- | ------------- | ----------------- |
| 1e-5       | 55.60%        | 68.69%            |
| 5e-5       | 54.44%        | 93.37%            |
| 1e-4       | 53.12%        | 96.87%            |
| 5e-4       | 48.91%        | 99.52%            |

### Key Observations

* Sparsity increases progressively with λ
* Accuracy degrades gradually, showing a clear trade-off
* At low λ (1e-5), accuracy improves over baseline (53.8% → 55.6%)

This suggests that **mild pruning acts as a regularizer**, similar to dropout, by removing redundant connections and improving generalization.

At higher λ values, the model achieves up to **~99.5% parameter sparsity** while retaining reasonable predictive performance.

---

## 5. Sparsity–Accuracy Trade-off

![Trade-off Graph](sparsity_vs_accuracy.png)

The model demonstrates a smooth transition from dense to highly sparse regimes.
This indicates that sparsity can be tuned precisely using λ.

---

## 6. Gate Distribution Analysis

![Gate Distribution](sparsity_distribution.png)

The distribution of gate values shows:

* A strong spike near 0 → most connections are pruned
* A smaller cluster away from 0 → important connections are preserved

This confirms that the network successfully identifies and retains only the most relevant weights.

---

## 7. Key Insights

* Self-pruning can be integrated directly into training using learnable gates
* L1 regularization on sigmoid gates effectively induces sparsity
* The pruning process exhibits **threshold-like behavior** at higher λ values
* Controlled tuning of λ enables a clear sparsity–accuracy trade-off
* Moderate sparsity can improve generalization performance

---

## 8. How to Run

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_network.py
```

Outputs:

* `results.json` (accuracy & sparsity metrics)
* `sparsity_distribution.png`
* `sparsity_vs_accuracy.png`

---

## 9. Conclusion

This work demonstrates a stable and effective self-pruning mechanism that learns compact neural architectures during training. The results highlight the practical trade-off between efficiency and performance, and show that pruning can serve both as a compression technique and a form of regularization.
