
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
BAND = 12
BAND_DIFF = 30
BAND_PEAK = 2
FROM_IT = 45
TO_IT = 60
BRANGE = (UB - LB + 1)

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

MIN_PEAKS = [1,  6, 14, 17, 20, 29, 35, 41, 49, 56, 62, 64, 70, 73, 76, 78, 83, 91]
MAX_PEAKS = [11, 16, 18, 24, 31, 39, 45, 51, 58, 63, 65, 72, 74, 77, 80, 86, 93]

# optimal bounds
#OPT_DAY_BOUNDS = list(map(int, "300,287,300,293,276,249,231,222,214,300,298,294,272,256,246,253,294,284,269,241,214,198,195,299,293,282,263,246,229,207,296,281,257,224,193,162,141,279,271,249,217,185,165,141,293,280,257,226,197,167,186,284,263,236,201,167,137,178,265,241,207,166,125,125,125,248,221,185,140,125,125,125,228,207,175,129,125,125,125,231,218,188,146,125,125,125,258,234,202,161,125,125,125,236,214,180,135,125,125,125".split(",")))

# prev 
#OPT_DAY_BOUNDS = list(map(int, "300,291,299,300,282,257,240,239,262,291,299,297,277,259,248,266,291,287,271,244,219,224,251,280,299,287,266,252,240,236,266,268,246,213,182,152,125,281,268,246,216,186,159,125,298,280,254,224,195,182,215,246,246,222,188,153,125,228,252,234,201,161,125,125,125,244,216,180,134,125,125,125,222,206,175,129,126,125,125,226,211,180,138,125,127,125,254,232,200,161,125,125,125,230,212,178,131,125,125,125".split(",")))
OPT_DAY_BOUNDS = list(map(int, "300,286,300,300,282,257,239,239,261,288,300,295,274,263,253,274,294,287,268,241,214,226,255,285,300,292,272,253,240,238,267,269,246,214,185,152,125,279,266,246,213,183,159,125,300,286,259,230,200,179,212,244,244,220,186,156,125,217,245,234,203,163,125,125,125,245,217,180,136,125,125,125,230,206,175,128,125,125,125,226,213,182,141,125,125,125,250,228,197,156,125,125,125,228,208,173,126,125,125,125".split(",")))

OPT_DAY_BOUNDS.append(OPT_DAY_BOUNDS[-1])

PENALTY = []
for i in range(len(OPT_DAY_BOUNDS) - 1):
    diff = abs(OPT_DAY_BOUNDS[i+1] - OPT_DAY_BOUNDS[i])
    penalty = (OPT_DAY_BOUNDS[i] - 125.0) / 400.0 * (OPT_DAY_BOUNDS[i] ** (0.5 + diff / 50.0))
    PENALTY.append(penalty)

#solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []

# constraint on sum(wij) = 1
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

# constraint on 125 <= sum(n_people_i * wij) <= 300
for i in range(len(groupedByDay)):
    vs = groupedByDay[i]

    bound = OPT_DAY_BOUNDS[i]

    if i < FROM_IT or i > TO_IT:
        cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))
        for (var, n_people) in vs:
            cons.SetCoefficient(var, n_people)
        continue

    cons = solver.Constraint(LB, UB)
    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

dayMatList = []
for i in range(DAYS):
    dayMat = []

    if i < FROM_IT or i > TO_IT:
        continue

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
    if len(dayMatList) > 0:
        prev = dayMatList[-1]
        cur = dayMat
        for d in range(BRANGE):
            cons = solver.Constraint(0, 0)
            for j in range(BRANGE):
                cons.SetCoefficient(prev[BRANGE * j + d], -1)
                cons.SetCoefficient(cur[BRANGE * d + j], 1)

    dayMatList.append(dayMat)

# objective function
objective = solver.Objective()
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        objective.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)


for i in range(len(dayMatList)):
    dayMat = dayMatList[i]
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            nday0 = d0 + LB
            nday1 = d1 + LB

            ndaydiff = 0 if FROM_IT + i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            var = dayMat[BRANGE * d0 + d1]

            objective.SetCoefficient(var, penalty)


objective.SetMinimization()

#solver.EnableOutput()
solver.SetNumThreads(4)
#solver.SetTimeLimit(10000)
solver.SetTimeLimit(400000)

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print("%s solution found" % ("feasible" if status == solver.FEASIBLE else "optimal"))
    print("solution: " + str(solver.Objective().Value()))

#    for i in range(len(familyVars)):
#        print(",".join([str(int(familyVars[i][j].solution_value())) for j in range(len(familyVars[i]))]))

    with open("submission_mip1.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [int(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

else:
    print(status)
    print("no success")
