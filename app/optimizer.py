"""Core optimization logic using PuLP for linear/mixed-integer programming."""

import pulp
from models import (
    Stockpile,
    OptimizationRequest,
    OptimizationResult,
    SelectedStockpile,
    CostBreakdown,
    AchievedChemistry,
)

# Tolerance for considering chemistry targets as "exact match"
EXACT_TOLERANCE = 0.01  # 0.01%


def optimize_blend(request: OptimizationRequest) -> OptimizationResult:
    """
    Optimize stockpile blending to meet tonnage and chemistry targets.

    Optimization modes:
    - distance: Minimize distance
    - material: Minimize cost
    - profit: Maximize profit (revenue - cost)
    """
    stockpiles = request.stockpiles
    target_tonnage = request.target_tonnage
    chemistry_targets = request.chemistry_targets
    min_increment = request.min_increment
    optimization_mode = request.optimization_mode

    # Validate total available tonnage
    total_available = sum(s.tonnage_available for s in stockpiles)
    if total_available < target_tonnage:
        return OptimizationResult(
            success=False,
            message=f"Insufficient tonnage: {total_available:,} wmt available, {target_tonnage:,} wmt required",
        )

    # Create the optimization problem
    # For profit mode, we maximize (which is minimize negative profit)
    if optimization_mode == "profit":
        prob = pulp.LpProblem("Stockpile_Blending", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("Stockpile_Blending", pulp.LpMinimize)

    # Decision variables: integer units of min_increment taken from each stockpile
    x = {
        s.name: pulp.LpVariable(
            f"x_{s.name}",
            lowBound=0,
            upBound=s.tonnage_available // min_increment,
            cat=pulp.LpInteger,
        )
        for s in stockpiles
    }
    # Actual tonnage is x[i] * min_increment
    tonnage = {s.name: x[s.name] * min_increment for s in stockpiles}

    # Build stockpile lookup
    stockpile_map = {s.name: s for s in stockpiles}

    # Calculate costs and revenue (treat None as 0)
    material_cost = pulp.lpSum(
        tonnage[s.name] * (s.cost_per_ton or 0) for s in stockpiles
    )
    distance_total = pulp.lpSum(
        tonnage[s.name] * (s.distance_km or 0) for s in stockpiles
    )
    total_revenue = pulp.lpSum(
        tonnage[s.name] * (s.revenue_per_ton or 0) for s in stockpiles
    )
    total_cost = material_cost
    profit = total_revenue - material_cost

    # Determine objective based on mode
    if optimization_mode == "distance":
        base_objective = distance_total
    elif optimization_mode == "material":
        base_objective = material_cost
    else:  # profit
        base_objective = profit

    # Chemistry penalty using auxiliary variables for absolute value
    elements = list(chemistry_targets.keys())

    # Large M for prioritizing chemistry compliance
    M_EXACT = 1000000
    M_APPROXIMATE = 10000

    chemistry_penalty = 0
    pos_vars = {}
    neg_vars = {}

    for element in elements:
        ct = chemistry_targets[element]
        target_pct = ct.target
        operator = ct.operator
        mode = ct.mode
        weight = ct.weight

        if weight <= 0:
            continue

        M = M_EXACT if mode == "exact" else M_APPROXIMATE

        weighted_sum = pulp.lpSum(
            tonnage[s.name] * s.chemistry.get(element, 0) for s in stockpiles
        )
        total_tons = pulp.lpSum(tonnage[s.name] for s in stockpiles)

        if operator == "range":
            # Two deviation pairs: one for min bound, one for max bound
            target_max_pct = ct.target_max if ct.target_max is not None else target_pct
            neg_min = pulp.LpVariable(f"neg_min_{element}", lowBound=0)
            pos_max = pulp.LpVariable(f"pos_max_{element}", lowBound=0)
            pos_min = pulp.LpVariable(f"pos_min_{element}", lowBound=0)
            neg_max = pulp.LpVariable(f"neg_max_{element}", lowBound=0)

            # deviation from min: weighted_sum - target_pct * total_tons = pos_min - neg_min
            prob += (
                weighted_sum - target_pct * total_tons == pos_min - neg_min,
                f"chemistry_deviation_min_{element}",
            )
            # deviation from max: weighted_sum - target_max_pct * total_tons = pos_max - neg_max
            prob += (
                weighted_sum - target_max_pct * total_tons == pos_max - neg_max,
                f"chemistry_deviation_max_{element}",
            )
            # Penalize below min (neg_min) and above max (pos_max)
            chemistry_penalty += M * weight * (neg_min + pos_max)
            # Store for reference
            neg_vars[element] = neg_min
            pos_vars[element] = pos_max
        else:
            pos_vars[element] = pulp.LpVariable(f"pos_{element}", lowBound=0)
            neg_vars[element] = pulp.LpVariable(f"neg_{element}", lowBound=0)

            prob += (
                weighted_sum - target_pct * total_tons == pos_vars[element] - neg_vars[element],
                f"chemistry_deviation_{element}",
            )

            if operator == "=":
                chemistry_penalty += M * weight * (pos_vars[element] + neg_vars[element])
            elif operator in (">", ">="):
                # Penalize only shortfall (neg)
                chemistry_penalty += M * weight * neg_vars[element]
            elif operator in ("<", "<="):
                # Penalize only excess (pos)
                chemistry_penalty += M * weight * pos_vars[element]

    # Set objective
    if optimization_mode == "profit":
        # Maximize profit, but subtract penalty (so higher penalty = worse)
        prob += base_objective - chemistry_penalty, "Objective"
    else:
        # Minimize cost + penalty
        prob += base_objective + chemistry_penalty, "Objective"

    # Constraint: meet target tonnage exactly
    prob += (
        pulp.lpSum(tonnage[s.name] for s in stockpiles) == target_tonnage,
        "Target_Tonnage",
    )

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus[status]
        return OptimizationResult(
            success=False,
            message=f"Optimization failed with status: {status_name}",
        )

    # Extract results
    selected = []
    total_material_cost = 0
    total_distance_weighted = 0
    total_revenue_val = 0

    for s in stockpiles:
        tons_taken = int(pulp.value(x[s.name]) * min_increment)

        if tons_taken > 0:
            mat_cost = tons_taken * (s.cost_per_ton or 0)
            rev = tons_taken * (s.revenue_per_ton or 0)
            prof = rev - mat_cost

            selected.append(SelectedStockpile(
                name=s.name,
                tonnage_taken=tons_taken,
                tonnage_available=s.tonnage_available,
                distance_km=s.distance_km,
                material_cost=round(mat_cost, 2),
                revenue=round(rev, 2),
                profit=round(prof, 2),
            ))

            total_material_cost += mat_cost
            total_distance_weighted += tons_taken * (s.distance_km or 0)
            total_revenue_val += rev

    total_cost_val = total_material_cost
    total_profit_val = total_revenue_val - total_cost_val
    actual_tonnage = sum(sp.tonnage_taken for sp in selected)
    distance_avg_km = total_distance_weighted / actual_tonnage if actual_tonnage > 0 else 0

    cost_breakdown = CostBreakdown(
        material_total=round(total_material_cost, 2),
        material_per_ton=round(total_material_cost / actual_tonnage, 2) if actual_tonnage > 0 else 0,
        cost_total=round(total_cost_val, 2),
        cost_per_ton=round(total_cost_val / actual_tonnage, 2) if actual_tonnage > 0 else 0,
        revenue_total=round(total_revenue_val, 2),
        revenue_per_ton=round(total_revenue_val / actual_tonnage, 2) if actual_tonnage > 0 else 0,
        profit_total=round(total_profit_val, 2),
        profit_per_ton=round(total_profit_val / actual_tonnage, 2) if actual_tonnage > 0 else 0,
        distance_avg_km=round(distance_avg_km, 2),
    )

    # Calculate achieved chemistry and generate recommendations
    achieved_chemistry = []
    recommendations = []
    has_exact_miss = False

    for element in elements:
        ct = chemistry_targets[element]
        target_pct = ct.target
        operator = ct.operator
        mode = ct.mode
        target_max_pct = ct.target_max

        weighted_sum = sum(
            sp.tonnage_taken * stockpile_map[sp.name].chemistry.get(element, 0)
            for sp in selected
        )
        achieved_pct = weighted_sum / actual_tonnage if actual_tonnage > 0 else 0
        deviation = achieved_pct - target_pct

        # Determine if constraint is satisfied
        if operator == "=":
            is_satisfied = abs(deviation) <= EXACT_TOLERANCE
        elif operator == ">":
            is_satisfied = achieved_pct > target_pct - EXACT_TOLERANCE
        elif operator == ">=":
            is_satisfied = achieved_pct >= target_pct - EXACT_TOLERANCE
        elif operator == "<":
            is_satisfied = achieved_pct < target_pct + EXACT_TOLERANCE
        elif operator == "<=":
            is_satisfied = achieved_pct <= target_pct + EXACT_TOLERANCE
        elif operator == "range":
            max_val = target_max_pct if target_max_pct is not None else target_pct
            is_satisfied = (achieved_pct >= target_pct - EXACT_TOLERANCE and
                           achieved_pct <= max_val + EXACT_TOLERANCE)
        else:
            is_satisfied = abs(deviation) <= EXACT_TOLERANCE

        achieved_chemistry.append(AchievedChemistry(
            element=element,
            operator=operator,
            target=target_pct,
            target_max=target_max_pct,
            achieved=round(achieved_pct, 4),
            deviation=round(deviation, 4),
            mode=mode,
            is_satisfied=is_satisfied,
        ))

        if mode == "exact" and not is_satisfied:
            has_exact_miss = True

    if has_exact_miss and min_increment > 1:
        recommendations.append(
            f"Some 'Exact' chemistry targets could not be met precisely. "
            f"Consider lowering the Minimum Blending Unit (currently {min_increment:,} wmt) "
            f"for better precision. Note: This may increase calculation time."
        )

    return OptimizationResult(
        success=True,
        message="Optimization completed successfully",
        selected_stockpiles=selected,
        cost_breakdown=cost_breakdown,
        achieved_chemistry=achieved_chemistry,
        total_tonnage=actual_tonnage,
        recommendations=recommendations,
    )
