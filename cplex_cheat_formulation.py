
import numpy as np
import pandas as pd
import cplex

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

DAYS = 100
LB = 125
UB = 300
BRANGE = UB - LB + 1

PREF_COST = 62868
ACC_COST = 6020.043432

fpath = 'family_data.csv'
families = pd.read_csv(fpath, index_col='family_id')

totPeople = 0
for i in range(families.shape[0]):
    totPeople += int(families.iloc[i]["n_people"])

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

tot_con = solver.linear_constraints.add(names = ["tot_con"])[0]
solver.linear_constraints.set_rhs(tot_con, PREF_COST)
solver.linear_constraints.set_senses(tot_con, "E")

for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        solver.linear_constraints.set_coefficients(tot_con, var, PENALTY1[j] + PENALTY2[j] * n_people)

dayList = []
costList = []
varList = []
for d0 in range(BRANGE):
    for d1 in range(BRANGE):
        nday0 = d0 + LB
        nday1 = d1 + LB

        ndaydiff = abs(nday0 - nday1)

        penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

        dayList.append(nday0)
        costList.append(penalty)
        varList.append(solver.variables.add(names = ["dvar_%d" % len(varList)])[0])
        solver.variables.set_types(varList[-1], solver.variables.type.integer)
        solver.variables.set_lower_bounds(varList[-1], 0)
        solver.variables.set_upper_bounds(varList[-1], DAYS)


day_con = solver.linear_constraints.add(names = ["day_con"])[0]
solver.linear_constraints.set_rhs(day_con, totPeople)
solver.linear_constraints.set_senses(day_con, "E")

for i in range(len(dayList)):
    solver.linear_constraints.set_coefficients(day_con, varList[i], dayList[i])

TOL = 0.001
acc_con = solver.linear_constraints.add(names = ["acc_con"])[0]
solver.linear_constraints.set_rhs(acc_con, ACC_COST - TOL)
solver.linear_constraints.set_range_values(acc_con, 2 * TOL)
solver.linear_constraints.set_senses(acc_con, "R")

for i in range(len(dayList)):
    solver.linear_constraints.set_coefficients(acc_con, varList[i], costList[i])


bOccupVars = []
for i in range(DAYS):
    vs = []
    for j in range(BRANGE):
        var = solver.variables.add(names = ["ddd_w%d_%d" % (i, j)])[0]
        solver.variables.set_types(var, solver.variables.type.binary)
        vs.append(var)
    bOccupVars.append(vs)

    eqb_con = solver.linear_constraints.add(names = ["eqb_con_%d" % i])[0]
    solver.linear_constraints.set_rhs(eqb_con, 1)
    solver.linear_constraints.set_senses(eqb_con, "E")

    for j, var in enumerate(vs):
        solver.linear_constraints.set_coefficients(eqb_con, var, 1)

    eq_con = solver.linear_constraints.add(names = ["eq_con_%d" % i])[0]
    solver.linear_constraints.set_rhs(eq_con, 0)
    solver.linear_constraints.set_senses(eq_con, "E")

    for j, var in enumerate(vs):
        solver.linear_constraints.set_coefficients(eq_con, var, LB + j)

    vs1 = groupedByDay[i]

    for (var, n_people) in vs1:
        solver.linear_constraints.set_coefficients(eq_con, var, -n_people)


for d0 in range(BRANGE):
    dcon = solver.linear_constraints.add(names = ["dcon__%d" % d0])[0]
    solver.linear_constraints.set_rhs(dcon, 0)
    solver.linear_constraints.set_senses(dcon, "E")

    for d1 in range(BRANGE):
        var = varList[d0][d1]
        solver.linear_constraints.set_coefficients(dcon, var, 1)

    for i in range(DAYS):
        solver.linear_constraints.set_coefficients(dcon, bOccupVars[i][d0], -1)

solver.objective.set_sense(solver.objective.sense.minimize)

solver.solve()

print("Solution status = ", solver.solution.get_status(), ":", end=' ')
print(solver.solution.status[solver.solution.get_status()])
print("Solution value  = ", solver.solution.get_objective_value())
