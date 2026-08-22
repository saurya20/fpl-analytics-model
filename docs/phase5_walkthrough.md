# Phase 5 Walkthrough - Weekly Transfer Optimization

This document details the design, constraints, FPL transfer rules, and example results of the Integer Linear Programming (ILP) FPL Weekly Transfer Optimizer.

---

## 1. Why Transfer Optimization is Different from Squad Optimization
* **Squad Optimization (Phase 4)**: Solves the FPL selection problem *from scratch*. It chooses 15 players out of the entire pool of 700+ active players to maximize starting XI points under budget and club constraints.
* **Transfer Optimization (Phase 5)**: Solves the FPL selection problem *subject to a starting state* (the current owned squad). Instead of selecting a squad from scratch, it decides which players to buy (transfers-in) and sell (transfers-out) to maximize net expected points, accounting for transfer penalties.

---

## 2. Why Only the Next Gameweek is Optimized
This optimizer is a **one-step-ahead (next gameweek) decision support tool**, not a multi-week recursive simulation planner. Designing it this way conforms to the production workflow:
1. Every week, new FPL data becomes available.
2. The pipeline is rerun: features are rebuilt, point-in-time LightGBM models are trained, and predictions are generated for the next upcoming gameweek only.
3. The transfer optimizer is run using the latest squad state and next-GW predictions to suggest the immediate transfer decision.
4. This avoids propagating prediction errors across a multi-week planning horizon.

---

## 3. Mathematical Formulation of Transfers

### Decision Variables
For each player $i \in \{1, ..., N\}$ in the prediction pool:
Let $U_i \in \{0, 1\}$ be the binary constant representing whether player $i$ is currently owned in the squad.
* $x_i \in \{0, 1\}$: $1$ if player $i$ is in the final squad, $0$ otherwise.
* $in_i \in \{0, 1\}$: $1$ if player $i$ is transferred in, $0$ otherwise.
* $out_i \in \{0, 1\}$: $1$ if player $i$ is transferred out, $0$ otherwise.
* $y \ge 0$: continuous auxiliary variable representing penalized transfers.

### Transfer Logic Constraints
1. **Final Squad Consistency**: $x_i = U_i + in_i - out_i$ for all $i$.
2. **Transfer In Limit**: $in_i \le 1 - U_i$ (can only transfer in if not already owned).
3. **Transfer Out Limit**: $out_i \le U_i$ (can only transfer out if already owned).
4. **Mutually Exclusive**: $in_i + out_i \le 1$ (cannot buy and sell the same player).

### Transfer Penalty Encoding
Let $F$ be the number of available free transfers (e.g. $F = 1$ or $2$), and let $H = 4.0$ be the configurable points hit penalty cost.
The penalized transfer variable $y$ is bounded by:

$$y \ge \sum_{i=1}^N in_i - F$$
$$y \ge 0$$

Because the objective function subtracts $H \times y$, the solver naturally minimizes $y$, ensuring that at optimality $y = \max(0, \sum in_i - F)$ (the number of transfers exceeding the free allowance).

### Objective Function
We maximize the Net Expected Points:

$$\text{Maximize } \sum_{i=1}^N r_i s_i + \sum_{i=1}^N r_i c_i + \epsilon \sum_{i=1}^N r_i v_i - H \times y$$

*(where $r_i$ is expected points, $s_i$ starter indicator, $c_i$ captain indicator, and $v_i$ vice-captain indicator)*

---

## 4. HOLD vs TRANSFER Strategy Comparison
The optimizer solves two separate strategies to determine the best course of action:
* **HOLD Strategy (Strictly 0 transfers)**: The squad is fixed to the current 15 owned players ($x_i = U_i$ for all $i$). Only the starting XI, captain, and vice-captain are optimized.
* **TRANSFER Strategy**: The solver is free to make transfers.
* **Comparison**: We calculate the net improvement:
  $$\Delta = \text{Transfer Net Expected Points} - \text{Hold Expected Points}$$
  If $\Delta > 0$, we recommend the transfers. Otherwise, we recommend **HOLD**.

---

## 5. Budget Constraints & Selling Value
* **Max Budget**: $C_{\text{squad}} + K$ (where $C_{\text{squad}}$ is the current squad value and $K$ is bank cash).
* **FPL Selling Price Limitation**: Under official FPL rules, players are sold at their purchase price plus 50% of the profit. Because our historical predictions do not track the user's specific purchase prices, we use a simplified budget model where selling price equals the player's current price. This is a documented limitation.

---

## 6. Example Optimizer Output: 2024-25 GW 15
Running:
```bash
python -m src.optimization.transfer_optimizer --season 2024-25 --gw 15 --squad data/input/current_squad.json --free-transfers 1
```

Yields the following mathematical solution:
* **Recommendation**: **TRANSFER**
* **Hold Expected Points**: 47.05
* **Transfer Net Expected Points**: 50.63
* **Net Improvement (Delta)**: **+3.59 points**

### Recommended Transfers:
* **OUT**: Ângelo Gabriel (MID, BOU, £4.5m) — Exp Pts: 0.00
* **IN**: Enzo Fernández (MID, CHE, £4.9m) — Exp Pts: 3.59
* **Transfers made**: 1 (Free: 1, Hits: 0)

### Starting XI (Post-Transfers):
* GK: Emiliano Martínez (AVL, £5.0m) — Exp Pts: 3.96
* DEF: Lucas Digne (AVL, £4.7m) — Exp Pts: 4.36
* DEF: Joško Gvardiol (MCI, £6.2m) — Exp Pts: 3.91
* DEF: Ashley Phillips (TOT, £4.0m) — Exp Pts: 0.00
* MID: Cole Palmer (CHE, £11.0m) — Exp Pts: 5.98 **(VC)**
* MID: Jarrod Bowen (WHU, £7.4m) — Exp Pts: 5.30
* MID: Enzo Fernández (CHE, £4.9m) — Exp Pts: 3.59
* MID: Bukayo Saka (ARS, £10.5m) — Exp Pts: 6.24 **(C)**
* MID: Andrey Santos (CHE, £4.5m) — Exp Pts: 0.00
* FWD: Ollie Watkins (AVL, £9.0m) — Exp Pts: 5.41
* FWD: Erling Haaland (MCI, £15.0m) — Exp Pts: 5.65

---

## 7. Data Leakage Prevention
Strict leakage guards prevent target columns (e.g., `target_points`, `target_minutes`) and actual gameweek performance fields (e.g., `total_points`, `bps`, `bonus`, `minutes`) from entering the optimizer. If detected, the shield raises a `ValueError("DATA LEAKAGE SHIELD TRIGGERED")`.

---

## 8. Limitations
1. **Simplified Selling Price**: Selling price is set to current price, ignoring purchase price profit taxes.
2. **Single-Period Focus**: Optimizes expected points for the next gameweek only, ignoring fixture difficulty trends across subsequent weeks (e.g., 3-5 GWs).
