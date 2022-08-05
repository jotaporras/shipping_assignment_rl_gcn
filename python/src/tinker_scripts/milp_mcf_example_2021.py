from ortools.linear_solver import pywraplp

if __name__ == "__main__":

    # Create the mip solver with the SCIP backend.
    # solver = pywraplp.Solver.CreateSolver("SCIP")
    solver = pywraplp.Solver("test_milp", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    b = {
        "b^1_1": 10,
        "b^1_2": 0,
        "b^1_3": 0,
        "b^1_4": 0,
        "b^1_5": -10,
        "b^2_1": 0,
        "b^2_2": 0,
        "b^2_3": 10,
        "b^2_4": 0,
        "b^2_5": -10,
    }
    # infinity = 10
    infinity = solver.Infinity()
    max_flow = 40  # all possible flow in one arc.
    x = {
        "x^1_12": solver.IntVar(0.0, infinity, "x^1_12"),
        "x^1_13": solver.IntVar(0.0, infinity, "x^1_13"),
        "x^1_25": solver.IntVar(0.0, infinity, "x^1_25"),
        "x^1_34": solver.IntVar(0.0, infinity, "x^1_34"),
        "x^1_45": solver.IntVar(0.0, infinity, "x^1_45"),
        "x^2_12": solver.IntVar(0.0, infinity, "x^2_12"),
        "x^2_13": solver.IntVar(0.0, infinity, "x^2_13"),
        "x^2_25": solver.IntVar(0.0, infinity, "x^2_25"),
        "x^2_34": solver.IntVar(0.0, infinity, "x^2_34"),
        "x^2_45": solver.IntVar(0.0, infinity, "x^2_45"),
    }

    # consolidation constraints.
    l1_25 = solver.IntVar(0.0, 1.0, "l^1_25")
    l2_25 = solver.IntVar(0.0, 1.0, "l^2_25")

    l1_45 = solver.IntVar(0.0, 1.0, "l^1_45")
    l2_45 = solver.IntVar(0.0, 1.0, "l^2_45")

    c = dict([(k, 1) for k in x.keys()])
    # c['x^2_45']=2

    # Constraints
    # Positivity
    # for k in x.keys():
    #     solver.Add(x[k]>=0)

    # Mass balance
    for k in [1, 2]:
        # for k in [1]:
        solver.Add(x[f"x^{k}_12"] + x[f"x^{k}_13"] == b[f"b^{k}_1"])
        solver.Add(-x[f"x^{k}_12"] + x[f"x^{k}_25"] == b[f"b^{k}_2"])
        solver.Add(-x[f"x^{k}_12"] + x[f"x^{k}_25"] == b[f"b^{k}_2"])
        solver.Add(-x[f"x^{k}_13"] + x[f"x^{k}_34"] == b[f"b^{k}_3"])
        solver.Add(-x[f"x^{k}_34"] + x[f"x^{k}_45"] == b[f"b^{k}_4"])
        solver.Add(-x[f"x^{k}_45"] - x[f"x^{k}_25"] == b[f"b^{k}_5"])

    # Linearized binary * flow constraints.
    solver.Add(x["x^1_25"] <= max_flow * l1_25)
    solver.Add(x["x^1_25"] >= l1_25)

    solver.Add(x["x^2_25"] <= max_flow * l2_25)
    solver.Add(x["x^2_25"] >= l2_25)

    solver.Add(x["x^2_45"] <= max_flow * l2_45)
    solver.Add(x["x^2_45"] >= l2_45)

    solver.Add(x["x^2_45"] <= max_flow * l2_45)
    solver.Add(x["x^2_45"] >= l2_45)

    # todo deleteme see if this is valid
    solver.Add(x["x^2_45"] + solver.Sum([]) >= l2_45)
    ####

    # todo maybe delete old constraints
    # Single destination constraint?
    # solver.Add(x["x^1_25"] <= (x["x^1_25"] * l1_25))
    # solver.Add(x["x^1_25"] <= (x["x^1_25"] * l1_25))
    # solver.Add(x["x^1_25"] >= l1_25)
    #
    # solver.Add(x["x^2_25"] <= (x["x^2_25"] * l2_25))
    # solver.Add(x["x^2_25"] >= l2_25)
    #
    # solver.Add(x["x^1_45"] <= (x["x^1_45"] * l1_45))
    # solver.Add(x["x^1_45"] >= l1_45)
    #
    # solver.Add(x["x^2_45"] <= (x["x^2_45"] * l2_45))
    # solver.Add(x["x^2_45"] >= l2_45)

    # solver.Add(l1_25 + l1_45 + l2_25 + l2_45 == 2)
    # Constraint all commodities must flow from same source
    # solver.Add(l1_25 == l2_25)
    # solver.Add(l1_45 == l2_45)
    solver.Add(l1_25 + l2_25 == 2 * l1_25)  # if you use one arc, use all K.
    solver.Add(l1_45 + l2_45 == 2 * l1_45)

    # Constraint must flow from one warehouse (i.e. sum of used arcs == k)
    # which arcs comprise this sum is enforced in the other constraint.
    # TODO: Is this better than doing separate constraints?
    solver.Add(l1_25 + l1_45 + l2_25 + l2_45 == 2)

    # objective
    first_key = list(x.keys())[0]
    obj = x[first_key] * c[first_key]
    for k in list(x.keys())[1:]:
        obj = obj + x[k] * c[k]

    solver.Minimize(obj)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        for k in list(x.keys()):
            # print(f"x[{k}] = {x[k].Value()}")
            print(f"x[{k}] = {x[k].solution_value()}")
        print(f"l^1_25 {l1_25.solution_value()}")
        print(f"l^2_25 {l2_25.solution_value()}")
        print(f"l^1_45 {l1_45.solution_value()}")
        print(f"l^2_45 {l2_45.solution_value()}")
        print("Objective value =", solver.Objective().Value())
        print("WallTime() =", solver.WallTime())
    else:
        print(status)
        print("The problem does not have an optimal solution.")
        print(solver)
