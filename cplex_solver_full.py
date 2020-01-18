
import numpy as np
import pandas as pd
import cplex

DAYS = 100
LB = 125
UB = 300
BRANGE = (UB - LB + 1)

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

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

dayMatList = []
for i in range(DAYS):
    dayMat = []

    # constraint on WMAT SUM
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            var = solver.variables.add(names = ["dmat_%d_%d_%d" % (i, d0, d1)])[0]
            solver.variables.set_types(var, solver.variables.type.binary)
            dayMat.append(var)

    con = solver.linear_constraints.add(names = ["con_dmat_%d" % i])[0]
    solver.linear_constraints.set_rhs(con, 1)
    solver.linear_constraints.set_senses(con, "E")
    for var in dayMat:
        solver.linear_constraints.set_coefficients(con, var, 1)

    # constraints on connection between CHOICE_W AND WMAT
    vs = groupedByDay[i]
    cons = solver.linear_constraints.add(names = ["con_choice_dmat_%d" % i])[0]
    solver.linear_constraints.set_rhs(cons, 0)
    solver.linear_constraints.set_senses(cons, "E")

    for (var, n_people) in vs:
        solver.linear_constraints.set_coefficients(cons, var, n_people)

    for ix, var in enumerate(dayMat):
        coef = LB + (ix // BRANGE)
        solver.linear_constraints.set_coefficients(cons, var, -coef)

    # constraint on row and next column
    if i > 0:
        prev = dayMatList[-1]
        cur = dayMat
        for d in range(BRANGE):
            cons = solver.linear_constraints.add(names = ["con_choice_dmat_%d_%d" % (i, d)])[0]
            solver.linear_constraints.set_rhs(cons, 0)
            solver.linear_constraints.set_senses(cons, "E")
            for j in range(BRANGE):
                solver.linear_constraints.set_coefficients(cons, prev[BRANGE * j + d], -1)
                solver.linear_constraints.set_coefficients(cons, cur[BRANGE * d + j], 1)

    dayMatList.append(dayMat)

for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        solver.objective.set_linear(var, PENALTY1[j] + PENALTY2[j] * n_people)

solver.objective.set_sense(solver.objective.sense.minimize)

solver.solve()

print("Solution status = ", solver.solution.get_status(), ":", end=' ')
print(solver.solution.status[solver.solution.get_status()])
print("Solution value  = ", solver.solution.get_objective_value())
