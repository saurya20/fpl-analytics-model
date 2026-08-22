import pulp
import pandas as pd

def get_optimization_objective(
    s: dict,
    c: dict,
    v: dict,
    df: pd.DataFrame,
    target_points_col: str = "predicted_points"
) -> pulp.LpAffineExpression:
    """
    Formulate the mathematical objective function:
      Maximize sum(expected_points_i * s_i) + sum(expected_points_i * c_i) + epsilon * sum(expected_points_i * v_i)
    
    The epsilon term (1e-4) acts as a tie-breaker that encourages the solver to choose the 
    second-best available starting player as vice-captain, without altering primary points optimization.
    """
    epsilon = 1e-4
    
    objective_terms = []
    for i in df.index:
        pts = df.loc[i, target_points_col]
        # Starters expected points
        objective_terms.append(pts * s[i])
        # Captain bonus expected points
        objective_terms.append(pts * c[i])
        # Deterministic vice-captain selection
        objective_terms.append(epsilon * pts * v[i])
        
    return pulp.lpSum(objective_terms)
