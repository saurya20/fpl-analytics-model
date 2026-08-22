# Phase 4 Walkthrough - FPL Squad Optimization

This document explains the mathematical design, decision variables, objective functions, constraints, and results of the Integer Linear Programming (ILP) FPL Squad Optimizer.

---

## 1. What is Integer Linear Programming (ILP)?
Integer Linear Programming (ILP) or Mixed Integer Linear Programming (MILP) is a mathematical optimization technique used to maximize or minimize a linear objective function subject to a set of linear equality and inequality constraints, where some or all of the decision variables are restricted to integer values (often binary $0$ or $1$).

In this project, we model player selection as binary decision variables ($1$ if selected, $0$ if not), and we use the open-source **PuLP** library in Python with the default **COIN-OR Branch-and-Cut (CBC)** solver to find the mathematically optimal squad.

---

## 2. Why is ILP appropriate for FPL?
FPL selection is a combinatorial optimization problem. Selecting 15 players out of 700+ active players while satisfying budget, position, and club limits yields an enormous search space (over $10^{30}$ possible combinations). 

A simple greedy heuristic (e.g. "select the highest expected points players") fails because:
* It does not guarantee satisfying the £100.0m budget constraint.
* It does not respect the maximum 3 players per Premier League club.
* It does not balance position requirements (e.g., exactly 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs).

ILP solves this search space to absolute mathematical optimality in less than 0.1 seconds, satisfying all constraints simultaneously.

---

## 3. Mathematical Formulation

### Decision Variables
For each player $i \in \{1, ..., N\}$ in the prediction pool:
* $x_i \in \{0, 1\}$: $1$ if player $i$ is selected in the 15-player squad, $0$ otherwise.
* $s_i \in \{0, 1\}$: $1$ if player $i$ is selected in the starting XI, $0$ otherwise.
* $c_i \in \{0, 1\}$: $1$ if player $i$ is selected as Captain, $0$ otherwise.
* $v_i \in \{0, 1\}$: $1$ if player $i$ is selected as Vice-Captain, $0$ otherwise.

### Objective Function
We maximize the expected points of the starting XI, incorporating the doubled points contribution of the captain:

$$\text{Maximize } \sum_{i=1}^N r_i s_i + \sum_{i=1}^N r_i c_i + \epsilon \sum_{i=1}^N r_i v_i$$

Where:
* $r_i$ is the expected points prediction for player $i$.
* $\epsilon = 10^{-4}$ is a tiny secondary tie-breaker that encourages the solver to designate the next-best starting player as vice-captain, without altering primary points optimization.

---

## 4. Constraint Encodings

### A. Squad Constraints
1. **Squad Size**: $\sum_{i=1}^N x_i = 15$
2. **Goalkeepers**: $\sum_{i=1}^N G_i x_i = 2$
3. **Defenders**: $\sum_{i=1}^N D_i x_i = 5$
4. **Midfielders**: $\sum_{i=1}^N M_i x_i = 5$
5. **Forwards**: $\sum_{i=1}^N F_i x_i = 3$
6. **Budget limit**: $\sum_{i=1}^N p_i x_i \le 100.0$ (where $p_i$ is the player's current price).
7. **Club limit**: For each club $T$, $\sum_{i=1}^N T_i x_i \le 3$.

### B. Starting XI Constraints
1. **Starters Size**: $\sum_{i=1}^N s_i = 11$
2. **Squad Consistency**: $s_i \le x_i$ for all $i$.
3. **Starting Goalkeeper**: $\sum_{i=1}^N G_i s_i = 1$
4. **Starting Defenders**: $\sum_{i=1}^N D_i s_i \ge 3$
5. **Starting Midfielders**: $\sum_{i=1}^N M_i s_i \ge 1$
6. **Starting Forwards**: $\sum_{i=1}^N F_i s_i \ge 1$

### C. Captaincy Constraints
1. **Captain is Starter**: $c_i \le s_i$ for all $i$.
2. **Vice-Captain is Starter**: $v_i \le s_i$ for all $i$.
3. **Uniqueness**: $c_i + v_i \le 1$ for all $i$.
4. **Captain Count**: $\sum_{i=1}^N c_i = 1$
5. **Vice-Captain Count**: $\sum_{i=1}^N v_i = 1$

---

## 5. Bench Ordering
After solving, the 4 unused squad players (where $x_i = 1$ and $s_i = 0$) are identified.
1. The backup Goalkeeper is placed on the GK bench slot.
2. The remaining 3 outfield substitutes are sorted descending by expected points and assigned indices $1, 2, 3$.

---

## 6. Double Gameweeks & Blank Gameweeks
* **Double Gameweeks**: DGW players naturally receive higher expected points when the LightGBM models predict multiple fixtures, making them more attractive to the solver.
* **Blank Gameweeks**: BGW players have 0 expected points. The solver naturally avoids them unless extremely tight budgets force selecting them as cheap bench warmers (e.g. a £3.9m defender with 0 expected points).

---

## 7. Data Leakage Prevention
To prevent scientific leakage, the optimizer runs entirely downstream of predictions. If it detects target actuals (e.g., `target_points`, `target_minutes`, `total_points`, `minutes`) in the input DataFrame, it raises a:

`DATA LEAKAGE SHIELD TRIGGERED`

value error, failing loudly.

---

## 8. Example Optimized Squad: 2024-25 GW 15
Running the command:
```bash
python -m src.optimization.squad_optimizer --season 2024-25 --gw 15
```

Yields the following mathematical solution:
* **Formation**: 3-5-2
* **Total Cost**: £99.8m (remaining budget £0.2m)
* **Expected Points**: 60.06

### Starting XI:
* **GK**: Kepa Arrizabalaga (BOU, £4.5m) — Exp Pts: 3.69
* **DEF**: Lucas Digne (AVL, £4.7m) — Exp Pts: 4.36
* **DEF**: Joško Gvardiol (MCI, £6.2m) — Exp Pts: 3.91
* **DEF**: Milos Kerkez (BOU, £4.6m) — Exp Pts: 3.75
* **MID**: Morgan Rogers (AVL, £5.3m) — Exp Pts: 5.10
* **MID**: Bukayo Saka (ARS, £10.5m) — Exp Pts: 6.24 (C)
* **MID**: Jarrod Bowen (WHU, £7.4m) — Exp Pts: 5.30
* **MID**: Cole Palmer (CHE, £11.0m) — Exp Pts: 5.98 (VC)
* **MID**: Justin Kluivert (BOU, £5.5m) — Exp Pts: 4.44
* **FWD**: Ollie Watkins (AVL, £9.0m) — Exp Pts: 5.41
* **FWD**: Erling Haaland (MCI, £15.0m) — Exp Pts: 5.65

### Bench:
* **GK Sub**: Marcus Bettinelli (CHE, £3.9m) — Exp Pts: 0.14
* **SUB 1**: James Bree (SOU, £3.9m) — Exp Pts: 0.98
* **SUB 2**: Charlie Taylor (SOU, £3.9m) — Exp Pts: 0.70
* **SUB 3**: Ross Stewart (SOU, £4.4m) — Exp Pts: 0.33

### Club Distribution:
* AVL: 3, BOU: 3, SOU: 3, CHE: 2, MCI: 2, ARS: 1, WHU: 1 (All $\le 3$).

---

## 9. Limitations
1. **Single Gameweek Horizon**: The current version does not account for transfer cost across weeks. A player with a good fixture this week might have terrible fixtures in the next 3 weeks. Future transfer optimizers will address this.
2. **Real-world Captaincy Risk**: If the captain does not play, the vice-captain receives the points. The model assumes captain starts, but does not explicitly weigh vice-captain points by probability of captain missing out.
