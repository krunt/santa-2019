
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
BAND = 6
BRANGE = (UB - LB + 1)
OPT_PREF_COST = 62868
OPT_ACC_COST = 6020

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

OPT_DAY_BOUNDS = list(map(int, "300,286,300,300,282,257,239,239,261,288,300,295,274,263,253,274,294,287,268,241,214,226,255,285,300,292,272,253,240,238,267,269,246,214,185,152,125,279,266,246,213,183,159,125,300,286,259,230,200,179,212,244,244,220,186,156,125,217,245,234,203,163,125,125,125,245,217,180,136,125,125,125,230,206,175,128,125,125,125,226,213,182,141,125,125,125,250,228,197,156,125,125,125,228,208,173,126,125,125,125".split(",")))

OPT_DAY_BOUNDS.append(OPT_DAY_BOUNDS[-1])

#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []

for i in range(data.shape[0]):
    days = data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5"]]

    n_people = int(data.iloc[i]["n_people"])

    vs = [] 
    for j, day in enumerate(days):
        vs.append(solver.NumVar(0, 1, "fam_w%d_%d" % (i, j)))
        groupedByDay[day - 1].append((vs[-1], n_people))

    cons = solver.Constraint(1, 1)
    for var in vs:
        cons.SetCoefficient(var, 1)

    familyVars.append(vs)

assert(len(groupedByDay) == DAYS)

for i in range(DAYS):
    vs = groupedByDay[i]

    #bound = OPT_DAY_BOUNDS[i]

    cons = solver.Constraint(LB, UB)
    #cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)


dayMatList = []
for i in range(DAYS):
    dayMat = []

    # constraint on WMAT SUM
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            dayMat.append(solver.NumVar(0, 1, "dmat_%d_%d_%d" % (i, d0, d1)))

    cons = solver.Constraint(1, 1)
    for var in dayMat:
        cons.SetCoefficient(var, 1)

    # constraints on connection between CHOICE_W AND WMAT
    vs = groupedByDay[i]
    cons = solver.Constraint(0, 0)

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

    for ix, var in enumerate(dayMat):
        coef = LB + (ix // BRANGE)
        cons.SetCoefficient(var, -coef)

    # constraint on row and next column
    if i > 0:
        prev = dayMatList[-1]
        cur = dayMat
        for d in range(BRANGE):
            cons = solver.Constraint(0, 0)
            for j in range(BRANGE):
                cons.SetCoefficient(prev[BRANGE * j + d], -1)
                cons.SetCoefficient(cur[BRANGE * d + j], 1)

    dayMatList.append(dayMat)


objective = solver.Objective()
objective.SetMinimization()

# preference cost
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        objective.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

# accounting cost
for i in range(DAYS):
    dayMat = dayMatList[i]
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            nday0 = d0 + LB
            nday1 = d1 + LB

            ndaydiff = 0 if i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            var = dayMat[BRANGE * d0 + d1]

            objective.SetCoefficient(var, penalty)


# opt pref cons
cons = solver.Constraint(OPT_PREF_COST, OPT_PREF_COST)
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        cons.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

# opt acc cons
cons = solver.Constraint(OPT_ACC_COST - 0.5, OPT_ACC_COST + 1)
for i in range(DAYS):
    dayMat = dayMatList[i]
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            nday0 = d0 + LB
            nday1 = d1 + LB

            ndaydiff = 0 if i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            var = dayMat[BRANGE * d0 + d1]

            cons.SetCoefficient(var, penalty)


solver.EnableOutput()
solver.SetNumThreads(4)
solver.SetTimeLimit(24 * 60 * 60 * 1000) # 24h

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print("%s solution found" % ("feasible" if status == solver.FEASIBLE else "optimal"))
    print("solution: " + str(solver.Objective().Value()))

    with open("submission_lp_big1.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [str(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%s" % (i, ",".join(lst)), file=fd)

else:
    print(status)
    print("no success")
