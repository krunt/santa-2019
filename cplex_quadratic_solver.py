
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

accVars = []
for i in range(DAYS):
    dVarList = []
    for j in range(BRANGE):
        var = solver.variables.add(names = ["acc_v%d_%d" % (i, j)])[0]
        solver.variables.set_types(var, solver.variables.type.binary)
        dVarList.append(var)

    accVars += dVarList

    con = solver.linear_constraints.add(names = ["ddd_eq_%d" % i])[0]
    solver.linear_constraints.set_rhs(con, 0)
    solver.linear_constraints.set_senses(con, "E")

    vs = groupedByDay[i]
    for (var, n_people) in vs:
        solver.linear_constraints.set_coefficients(con, var, -n_people)

    for j in range(BRANGE):
        solver.linear_constraints.set_coefficients(con, dVarList[j], LB + j)

    con = solver.linear_constraints.add(names = ["done_eq_%d" % i])[0]
    solver.linear_constraints.set_rhs(con, 1)
    solver.linear_constraints.set_senses(con, "E")

    for j in range(BRANGE):
        solver.linear_constraints.set_coefficients(con, dVarList[j], 1)


for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        solver.objective.set_linear(var, PENALTY1[j] + PENALTY2[j] * n_people)

def get_acc_penalty(day0, day1, nday0, nday1):
    ndaydiff = 0 if day1 == DAYS else abs(nday0 - nday1)
    return (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

for day in range(1, DAYS):
    day0 = day - 1
    day1 = day

    for i in range(BRANGE):
        for j in range(i, BRANGE):
            nday0 = LB + i
            nday1 = LB + j
            penalty = get_acc_penalty(day0, day1, nday0, nday1)
            var0 = accVars[day0 * BRANGE + i]
            var1 = accVars[day1 * BRANGE + j]
            solver.objective.set_quadratic_coefficients(var0, var1, penalty)

solver.objective.set_sense(solver.objective.sense.minimize)

solver.solve()

print("Solution status = ", solver.solution.get_status(), ":", end=' ')
print(solver.solution.status[solver.solution.get_status()])
print("Solution value  = ", solver.solution.get_objective_value())
