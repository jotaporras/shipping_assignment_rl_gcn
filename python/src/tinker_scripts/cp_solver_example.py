"""
A test to understand how to solve CP probelems using Google OR Tools.
"""
from __future__ import print_function
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

# TODO aquí quedé, convertir esto en service cod.e
model = cp_model.CpModel()
# Create the mip solver with the CBC backend.
# solver = pywraplp.Solver(
#     "simple_mip_program", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
# )
# infinity = solver.infinity()

# Create variables
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
infinity = 10
x = {
    "x^1_12": model.NewIntVar(0, infinity, "x^1_12"),
    "x^1_13": model.NewIntVar(0, infinity, "x^1_13"),
    "x^1_25": model.NewIntVar(0, infinity, "x^1_25"),
    "x^1_34": model.NewIntVar(0, infinity, "x^1_34"),
    "x^1_45": model.NewIntVar(0, infinity, "x^1_45"),
    "x^2_12": model.NewIntVar(0, infinity, "x^2_12"),
    "x^2_13": model.NewIntVar(0, infinity, "x^2_13"),
    "x^2_25": model.NewIntVar(0, infinity, "x^2_25"),
    "x^2_34": model.NewIntVar(0, infinity, "x^2_34"),
    "x^2_45": model.NewIntVar(0, infinity, "x^2_45"),
}

# consolidation constraints.
l1_25 = model.NewBoolVar("l^1_25")
l2_25 = model.NewBoolVar("l^2_25")

l1_45 = model.NewBoolVar("l^1_45")
l2_45 = model.NewBoolVar("l^2_45")

c = dict([(k, 1) for k in x.keys()])
# c['x^2_45']=2

# Constraints
# Positivity
# for k in x.keys():
#     solver.Add(x[k]>=0)

# Mass balance
for k in [1, 2]:
    # for k in [1]:
    model.Add(x[f"x^{k}_12"] + x[f"x^{k}_13"] == b[f"b^{k}_1"])
    model.Add(-x[f"x^{k}_12"] + x[f"x^{k}_25"] == b[f"b^{k}_2"])
    model.Add(-x[f"x^{k}_12"] + x[f"x^{k}_25"] == b[f"b^{k}_2"])
    model.Add(-x[f"x^{k}_13"] + x[f"x^{k}_34"] == b[f"b^{k}_3"])
    model.Add(-x[f"x^{k}_34"] + x[f"x^{k}_45"] == b[f"b^{k}_4"])
    model.Add(-x[f"x^{k}_45"] - x[f"x^{k}_25"] == b[f"b^{k}_5"])


model.Add(x["x^1_25"] > 0).OnlyEnforceIf(l1_25)
model.Add(x["x^1_25"] == 0).OnlyEnforceIf(l1_25.Not())

model.Add(x["x^2_25"] > 0).OnlyEnforceIf(l2_25)
model.Add(x["x^2_25"] == 0).OnlyEnforceIf(l2_25.Not())
#
model.Add(x["x^1_45"] > 0).OnlyEnforceIf(l1_45)
model.Add(x["x^1_45"] == 0).OnlyEnforceIf(l1_45.Not())

model.Add(x["x^2_45"] > 0).OnlyEnforceIf(l2_45)
model.Add(x["x^2_45"] == 0).OnlyEnforceIf(l2_45.Not())

#
# model.AddBoolOr([l1_45,l2_45])
model.Add(l1_25 + l1_45 + l2_25 + l2_45 == 2)
model.Add(l1_25 == l2_25)
model.Add(l1_45 == l2_45)
# #model.Add(l1_25 * x["x^1_25"] == x["x^1_25"])
# # solver.Add(l2_25*x["x^2_25"] == x["
# # solver.Add(l1_45*x["x^1_45"] == x["x^1_45"])
# solver.Add(l2_45*x["x^2_45"] == x["x^2_45"])


# objective
first_key = list(x.keys())[0]
obj = x[first_key] * c[first_key]
for k in list(x.keys())[1:]:
    obj = obj + x[k] * c[k]

model.Minimize(obj)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print("Solution:")
    for k in list(x.keys()):
        # print(f"x[{k}] = {x[k].Value()}")
        print(f"x[{k}] = {solver.Value(x[k])}")
    print(f"l^1_25 {solver.Value(l1_25)}")
    print(f"l^2_25 {solver.Value(l2_25)}")
    print(f"l^1_45 {solver.Value(l1_45)}")
    print(f"l^2_45 {solver.Value(l2_45)}")
    print("Objective value =", solver.ObjectiveValue())
    print("UserTime =", solver.UserTime())
    print("WallTime() =", solver.WallTime())
else:
    print(status)
    print("The problem does not have an optimal solution.")
    print(model)
