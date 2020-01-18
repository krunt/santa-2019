
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
BAND = 4
BAND_DIFF = 30
BAND_PEAK = 2

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

MIN_PEAKS = [1,  6, 14, 17, 20, 29, 35, 41, 49, 56, 62, 64, 70, 73, 76, 78, 83, 91]
MAX_PEAKS = [11, 16, 18, 24, 31, 39, 45, 51, 58, 63, 65, 72, 74, 77, 80, 86, 93]

# optimal bounds
OPT_DAY_BOUNDS = list(map(int, "300,287,300,293,276,249,231,222,214,300,298,294,272,256,246,253,294,284,269,241,214,198,195,299,293,282,263,246,229,207,296,281,257,224,193,162,141,279,271,249,217,185,165,141,293,280,257,226,197,167,186,284,263,236,201,167,137,178,265,241,207,166,125,125,125,248,221,185,140,125,125,125,228,207,175,129,125,125,125,231,218,188,146,125,125,125,258,234,202,161,125,125,125,236,214,180,135,125,125,125".split(",")))

# prev 
#OPT_DAY_BOUNDS = list(map(int, "300,291,299,300,282,257,240,239,262,291,299,297,277,259,248,266,291,287,271,244,219,224,251,280,299,287,266,252,240,236,266,268,246,213,182,152,125,281,268,246,216,186,159,125,298,280,254,224,195,182,215,246,246,222,188,153,125,228,252,234,201,161,125,125,125,244,216,180,134,125,125,125,222,206,175,129,126,125,125,226,211,180,138,125,127,125,254,232,200,161,125,125,125,230,212,178,131,125,125,125".split(",")))
#OPT_DAY_BOUNDS = list(map(int, "300,285,300,300,284,261,244,245,263,290,298,298,274,259,250,268,292,289,271,244,220,225,253,282,298,288,269,252,240,237,264,268,246,214,182,152,125,281,267,243,210,178,154,125,300,283,259,229,200,175,212,246,246,219,185,153,125,222,248,236,203,163,125,125,125,246,219,183,139,125,125,125,231,208,175,128,125,125,125,227,213,182,138,125,126,125,253,230,197,156,125,125,125,229,208,172,126,125,125,125".split(",")))

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

fpath = 'submission_lp_big.csv'
submOptimal = pd.read_csv(fpath, index_col='family_id')

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

# add optimal inertia constraints
varsDefined = 0
for i in range(data.shape[0]):
    optVals = submOptimal.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"]]

    for j, optVal in enumerate(optVals):
        if int(optVal) == 1:
            cons = solver.Constraint(1, 1)
            cons.SetCoefficient(familyVars[i][j], 1)
            varsDefined += 1
            break

print("vars defined/total (%d/%d)" % (varsDefined, data.shape[0]))

# constraint on 125 <= sum(n_people_i * wij) <= 300
for i in range(len(groupedByDay)):
    vs = groupedByDay[i]

    bound = OPT_DAY_BOUNDS[i]

    cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)


# objective function
objective = solver.Objective()
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        objective.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

objective.SetMinimization()

#solver.EnableOutput()
solver.SetNumThreads(4)
#solver.SetTimeLimit(10000)
solver.SetTimeLimit(100000)

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print("%s solution found" % ("feasible" if status == solver.FEASIBLE else "optimal"))
    print("solution: " + str(solver.Objective().Value()))

#    for i in range(len(familyVars)):
#        print(",".join([str(int(familyVars[i][j].solution_value())) for j in range(len(familyVars[i]))]))

    with open("submission_mip2.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [int(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

else:
    print(status)
    print("no success")
