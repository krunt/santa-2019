
import numpy as np
import pandas as pd
import cplex

DAYS = 100
LB = 125
UB = 300

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

OPT_DAY_BOUNDS = list(map(int, "300,286,300,300,282,257,239,239,261,288,300,295,274,263,253,274,294,287,268,241,214,226,255,285,300,292,272,253,240,238,267,269,246,214,185,152,125,279,266,246,213,183,159,125,300,286,259,230,200,179,212,244,244,220,186,156,125,217,245,234,203,163,125,125,125,245,217,180,136,125,125,125,230,206,175,128,125,125,125,226,213,182,141,125,125,125,250,228,197,156,125,125,125,228,208,173,126,125,125,125".split(",")))

OPT_DAY_BOUNDS.append(OPT_DAY_BOUNDS[-1])

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

solver = cplex.Cplex()

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []

for i in range(data.shape[0]):
    days = data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"]]

    n_people = int(data.iloc[i]["n_people"])

    vs = [] 
    for j, day in enumerate(days):
        vs.append(solver.variables.add(names = ["fam_w%d_%d" % (i, j)])[0])
        solver.variables.set_types(vs[-1], solver.variables.type.binary)
        groupedByDay[day - 1].append((vs[-1], n_people))

    con = solver.linear_constraints.add(names = ["con_ch_%d" % i])[0]
    solver.linear_constraints.set_rhs(con, 1)
    solver.linear_constraints.set_senses(con, "E")

    for var in vs:
        solver.linear_constraints.set_coefficients(con, var, 1)

    familyVars.append(vs)

assert(len(groupedByDay) == DAYS)

for i in range(DAYS):
    vs = groupedByDay[i]

    con = solver.linear_constraints.add(names = ["con_lbub_%d" % i])[0]
    solver.linear_constraints.set_rhs(con, LB)
    solver.linear_constraints.set_range_values(con, UB - LB)
    solver.linear_constraints.set_senses(con, "R")

    for (var, n_people) in vs:
        solver.linear_constraints.set_coefficients(con, var, n_people)

for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        solver.objective.set_linear(var, PENALTY1[j] + PENALTY2[j] * n_people)

solver.objective.set_sense(solver.objective.sense.minimize)

solver.solve()

print("Solution status = ", solver.solution.get_status(), ":", end=' ')
print(solver.solution.status[solver.solution.get_status()])
print("Solution value  = ", solver.solution.get_objective_value())
