import pulp
import pandas as pd

def add_squad_size_constraint(prob: pulp.LpProblem, x: dict, df: pd.DataFrame):
    """Exactly 15 players must be selected in the squad."""
    prob += pulp.lpSum(x[i] for i in df.index) == 15

def add_position_constraints(prob: pulp.LpProblem, x: dict, df: pd.DataFrame):
    """Exactly 2 GKs, 5 DEFs, 5 MIDs, and 3 FWDs in the 15-player squad."""
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "GK") == 2
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "DEF") == 5
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "MID") == 5
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "FWD") == 3

def add_budget_constraint(prob: pulp.LpProblem, x: dict, df: pd.DataFrame, budget: float = 100.0):
    """Total cost of the 15-player squad cannot exceed the budget."""
    prob += pulp.lpSum(df.loc[i, "current_price"] * x[i] for i in df.index) <= budget

def add_club_constraints(prob: pulp.LpProblem, x: dict, df: pd.DataFrame):
    """Maximum of 3 players from any single Premier League club."""
    unique_teams = df["team"].dropna().unique()
    for team in unique_teams:
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "team"] == team) <= 3

def add_starting_xi_constraints(prob: pulp.LpProblem, x: dict, s: dict, df: pd.DataFrame):
    """Exactly 11 starters, who must belong to the selected 15-player squad."""
    for i in df.index:
        prob += s[i] <= x[i]
    prob += pulp.lpSum(s[i] for i in df.index) == 11

def add_starter_position_constraints(prob: pulp.LpProblem, s: dict, df: pd.DataFrame):
    """Starters must have exactly 1 GK, at least 3 DEFs, at least 1 MID, and at least 1 FWD."""
    prob += pulp.lpSum(s[i] for i in df.index if df.loc[i, "position"] == "GK") == 1
    prob += pulp.lpSum(s[i] for i in df.index if df.loc[i, "position"] == "DEF") >= 3
    prob += pulp.lpSum(s[i] for i in df.index if df.loc[i, "position"] == "MID") >= 1
    prob += pulp.lpSum(s[i] for i in df.index if df.loc[i, "position"] == "FWD") >= 1

def add_captaincy_constraints(prob: pulp.LpProblem, s: dict, c: dict, v: dict, df: pd.DataFrame):
    """Captain and vice-captain must be unique starting players."""
    for i in df.index:
        prob += c[i] <= s[i]
        prob += v[i] <= s[i]
        prob += c[i] + v[i] <= 1
        
    prob += pulp.lpSum(c[i] for i in df.index) == 1
    prob += pulp.lpSum(v[i] for i in df.index) == 1
