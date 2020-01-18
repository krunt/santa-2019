
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
BAND = 8
BAND_DIFF = 30
BAND_PEAK = 2
OPT_PREF_COST = 62868

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

MIN_PEAKS = [1,  6, 14, 17, 20, 29, 35, 41, 49, 56, 62, 64, 70, 73, 76, 78, 83, 91]
MAX_PEAKS = [11, 16, 18, 24, 31, 39, 45, 51, 58, 63, 65, 72, 74, 77, 80, 86, 93]

# optimal bounds
#OPT_DAY_BOUNDS = list(map(int, "300,292,300,292,275,248,231,217,210,298,296,291,270,254,248,254,297,288,271,242,214,196,199,298,291,277,257,237,213,186,292,279,254,223,189,158,142,283,269,243,211,179,170,141,294,278,254,224,193,163,186,287,267,241,206,173,140,182,265,243,209,168,125,125,125,252,226,192,150,125,125,125,230,208,177,133,125,125,125,231,216,190,149,125,125,125,259,236,204,166,127,125,125,240,219,186,141,125,125,125".split(",")))

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
#solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

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

lowerBounds = []
upperBounds = []

# constraint on 125 <= sum(n_people_i * wij) <= 300
for i in range(len(groupedByDay)):
    vs = groupedByDay[i]

    #cons = solver.Constraint(LB, UB)

    bound = OPT_DAY_BOUNDS[i]

#    ixmin = -1
#    for j in range(len(MIN_PEAKS)):
#        if i >= MIN_PEAKS[j] - BAND_PEAK and i <= MIN_PEAKS[j] + BAND_PEAK:
#            ixmin = j
#            break
#    ixmax = -1
#    for j in range(len(MAX_PEAKS)):
#        if i >= MAX_PEAKS[j] - BAND_PEAK and i <= MAX_PEAKS[j] + BAND_PEAK:
#            ixmax = j
#            break
#
#    if ixmin != -1:
#        cons = solver.Constraint(max(bound, LB), min(bound + BAND_DIFF, UB))
#        for (var, n_people) in vs:
#            cons.SetCoefficient(var, n_people)
#        continue
#
#    if ixmax != -1:
#        cons = solver.Constraint(max(bound - BAND_DIFF, LB), min(bound, UB))
#        for (var, n_people) in vs:
#            cons.SetCoefficient(var, n_people)
#        continue

    #cons = solver.Constraint(LB, UB)

    #ratio = penalty / np.max(PENALTY)
    #ratio = (bound - LB) / (UB - LB)
    #band = round(BAND + ratio * BAND)

    lowerBounds.append(max(LB, bound - BAND))
    upperBounds.append(min(UB, bound + BAND))

    cons = solver.Constraint(lowerBounds[-1], upperBounds[-1])

    #cons = solver.Constraint(LB, UB)

#    cons = None
#    if (i > 45 and i < 55) or (i < 34):
#        cons = solver.Constraint(LB, UB)
#    else:
#        cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))
    #cons = solver.Constraint(max(LB, bound - BAND), min(UB, bound + BAND))

    #cons = solver.Constraint(LB, UB)

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

#absVars = []
#for i in range(1, len(groupedByDay)):
#    vs0 = groupedByDay[i-1]
#    vs1 = groupedByDay[i]
#
#    d0 = upperBounds[i-1] - lowerBounds[i-1]
#    d1 = upperBounds[i] - lowerBounds[i]
#
#    v0 = solver.IntVar(0, d0, "vid_%d_0" % i)
#    v1 = solver.IntVar(0, d1, "vid_%d_1" % i)
#
#    cons0 = solver.Constraint(lowerBounds[i-1], lowerBounds[i-1])
#    for (var, n_people) in vs0:
#        cons0.SetCoefficient(var, n_people)
#    cons0.SetCoefficient(v0, -1)
#
#    cons1 = solver.Constraint(lowerBounds[i], lowerBounds[i])
#    for (var, n_people) in vs1:
#        cons1.SetCoefficient(var, n_people)
#    cons1.SetCoefficient(v1, -1)
#
#    dd0 = solver.IntVar(0, 1, "vidb_%d_0" % i)
#    dd1 = solver.IntVar(0, 1, "vidb_%d_1" % i)
#    cons = solver.Constraint(1, 1)
#    cons.SetCoefficient(dd0, 1)
#    cons.SetCoefficient(dd1, 1)
#
#    mx = max(d0, d1)
#    yVar = solver.IntVar(0, mx, "vary_%d" % i)
#
#    cons0 = solver.Constraint(0, 100000)
#    cons0.SetCoefficient(dd0, 2 * mx)
#    cons0.SetCoefficient(yVar, -1)
#    cons0.SetCoefficient(v0, 1)
#    cons0.SetCoefficient(v1, -1)
#
#    cons1 = solver.Constraint(0, 100000)
#    cons1.SetCoefficient(dd1, 2 * mx)
#    cons1.SetCoefficient(yVar, -1)
#    cons1.SetCoefficient(v1, 1)
#    cons1.SetCoefficient(v0, -1)
#
#    yVar1 = solver.IntVar(0, 2*mx, "vary1_%d" % i)
#    cons = solver.Constraint(abs(lowerBounds[i] - lowerBounds[i-1]), abs(lowerBounds[i] - lowerBounds[i-1]))
#    cons.SetCoefficient(yVar1, 1)
#    cons.SetCoefficient(yVar, -1)
#
#    absVars.append(yVar1)


#for i in range(len(groupedByDay) - 1):
#    vs0 = groupedByDay[i]
#    vs1 = groupedByDay[i+1]
#
#    bound0 = OPT_DAY_BOUNDS[i]
#    bound1 = OPT_DAY_BOUNDS[i+1]
#
#    if abs(bound0 - bound1) > BAND_DIFF:
#        continue
#
#    cons = solver.Constraint(-BAND_DIFF, BAND_DIFF)
#
#    for (var, n_people) in vs0:
#        cons.SetCoefficient(var, n_people)
#    for (var, n_people) in vs1:
#        cons.SetCoefficient(var, -n_people)

#cons = solver.Constraint(OPT_PREF_COST, OPT_PREF_COST + 4000)
#for i in range(data.shape[0]):
#    n_people = int(data.iloc[i]["n_people"])
#
#    for j, var in enumerate(familyVars[i]):
#        cons.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

# objective function
objective = solver.Objective()
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        objective.SetCoefficient(var, PENALTY1[j] + PENALTY2[j] * n_people)

#for var in absVars:
#    objective.SetCoefficient(var, 1)

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

    with open("submission_mip1.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [int(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

else:
    print(status)
    print("no success")
