
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
BAND = 20
BRANGE = (UB - LB + 1)
NACC = BRANGE
BPART = BRANGE // NACC
BESTSCORE = 69000

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

#OPT_DAY_BOUNDS = list(map(int, "300,287,300,293,276,249,231,222,214,300,298,294,272,256,246,253,294,284,269,241,214,198,195,299,293,282,263,246,229,207,296,281,257,224,193,162,141,279,271,249,217,185,165,141,293,280,257,226,197,167,186,284,263,236,201,167,137,178,265,241,207,166,125,125,125,248,221,185,140,125,125,125,228,207,175,129,125,125,125,231,218,188,146,125,125,125,258,234,202,161,125,125,125,236,214,180,135,125,125,125".split(",")))

OPT_DAY_BOUNDS = list(map(int, "300,286,300,300,282,257,239,239,261,288,300,295,274,263,253,274,294,287,268,241,214,226,255,285,300,292,272,253,240,238,267,269,246,214,185,152,125,279,266,246,213,183,159,125,300,286,259,230,200,179,212,244,244,220,186,156,125,217,245,234,203,163,125,125,125,245,217,180,136,125,125,125,230,206,175,128,125,125,125,226,213,182,141,125,125,125,250,228,197,156,125,125,125,228,208,173,126,125,125,125".split(",")))

OPT_DAY_BOUNDS.append(OPT_DAY_BOUNDS[-1])

#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver = pywraplp.Solver('glpk_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []

for i in range(data.shape[0]):
    days = data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"]]

    n_people = int(data.iloc[i]["n_people"])

    vs = [] 
    for j, day in enumerate(days):
        vs.append(solver.IntVar(0, 1, "fam_w%d_%d" % (i, j)))
        groupedByDay[day - 1].append((vs[-1], n_people))

    cons = solver.Constraint(1, 1)
    for var in vs:
        cons.SetCoefficient(var, 1)

    familyVars.append(vs)

assert(len(groupedByDay) == DAYS)

for i in range(DAYS):
    vs = groupedByDay[i]

    #cons = solver.Constraint(LB, UB)

    bound = OPT_DAY_BOUNDS[i]

    #cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))
    cons = solver.Constraint(LB, UB)

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

dayMatList = []
for i in range(DAYS):
    dayMat = []

    # constraint on WMAT SUM
    for d0 in range(NACC):
        for d1 in range(NACC):
            dayMat.append(solver.IntVar(0, 1, "dmat_%d_%d_%d" % (i, d0, d1)))

    cons = solver.Constraint(1, 1)
    for var in dayMat:
        cons.SetCoefficient(var, 1)

    vs = groupedByDay[i]

    # constraints on connection between CHOICE_W AND WMAT
    cons = solver.Constraint(0, 0)
    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people / BPART)

    for ix, var in enumerate(dayMat):
        coef = (LB + (ix // NACC) * BPART) / BPART
        cons.SetCoefficient(var, -coef)

    # constraint on row and next column
    if i > 0:
        prev = dayMatList[-1]
        cur = dayMat
        for d in range(NACC):
            cons = solver.Constraint(0, 0)
            for j in range(NACC):
                cons.SetCoefficient(prev[NACC * j + d], -1)
                cons.SetCoefficient(cur[NACC * d + j], 1)

    dayMatList.append(dayMat)


objective = solver.Objective()
objective.SetMinimization()

# preference cost
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        objective.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

cnt = 0

# accounting cost
for i in range(DAYS):
    dayMat = dayMatList[i]
    for d0 in range(NACC):
        for d1 in range(NACC):
            nday0 = d0 * BPART + LB
            nday1 = d1 * BPART + LB

            ndaydiff = 0 if i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            var = dayMat[NACC * d0 + d1]

            #objective.SetCoefficient(var, penalty)
            penalty = round(penalty, 2)

            if penalty > BESTSCORE:
                cons = solver.Constraint(0, 0)
                cons.SetCoefficient(var, 1)
                cnt += 1
                continue

            objective.SetCoefficient(var, penalty)
                
print("ignored %d vars" % cnt)

#solver.EnableOutput()
#solver.SetNumThreads(8)
solver.SetTimeLimit(5 * 24 * 60 * 60 * 1000) # 24h
#solver.SetTimeLimit(800000) # 24h

#with open("milp_santa.lp", "w") as fd:
#    print(solver.ExportModelAsLpFormat(False), file=fd)
#
#with open("milp_santa.mps", "w") as fd:
#    print(solver.ExportModelAsMpsFormat(False), file=fd)

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print("%s solution found" % ("feasible" if status == solver.FEASIBLE else "optimal"))
    print("solution: " + str(solver.Objective().Value()))

    with open("submission_mip_big_scip.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [int(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

else:
    print(status)
    print("no success")
