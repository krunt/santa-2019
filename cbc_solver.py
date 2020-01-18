
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DAYS = 100
LB = 125
UB = 300
#BAND = 128
BRANGE = (UB - LB + 1)
OPT_ACC_COST = 6200

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

#OPT_DAY_BOUNDS_OPT = list(map(int, "300,292,300,292,275,248,231,217,210,298,296,291,270,254,248,254,297,288,271,242,214,196,199,298,291,277,257,237,213,186,292,279,254,223,189,158,142,283,269,243,211,179,170,141,294,278,254,224,193,163,186,287,267,241,206,173,140,182,265,243,209,168,125,125,125,252,226,192,150,125,125,125,230,208,177,133,125,125,125,231,216,190,149,125,125,125,259,236,204,166,127,125,125,240,219,186,141,125,125,125".split(",")))

OPT_DAY_BOUNDS_OPT = list(map(int, "300,288,300,300,281,256,239,243,267,294,300,300,278,263,256,278,300,293,274,247,223,229,255,283,299,288,269,253,245,243,271,273,249,215,184,152,125,280,266,243,210,178,154,125,300,281,256,226,195,161,125,284,262,235,201,169,137,125,265,240,206,165,125,125,125,248,219,183,139,125,125,125,225,206,173,126,125,125,125,226,212,180,136,125,125,125,253,232,200,159,125,125,125,228,207,173,126,125,125,125".split(",")))

OPT_DAY_BANDS = []

#for (v0, v1) in zip(OPT_DAY_BOUNDS_OPT, OPT_DAY_BOUNDS):
    #OPT_DAY_BANDS.append(max(2 * abs(v0 - v1), 8))

for i in range(len(OPT_DAY_BOUNDS_OPT)):
    #if i < DAYS // 2:
        #OPT_DAY_BANDS.append(88)
    #else:
        #OPT_DAY_BANDS.append(56)
    #OPT_DAY_BANDS.append(48)
    OPT_DAY_BANDS.append(BRANGE)

OPT_DAY_BOUNDS = OPT_DAY_BOUNDS_OPT
OPT_DAY_BOUNDS.append(OPT_DAY_BOUNDS[-1])
OPT_DAY_BANDS.append(OPT_DAY_BANDS[-1])

#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
#solver = pywraplp.Solver('glpk_program', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
solver = pywraplp.Solver('glpk_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

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
    band = OPT_DAY_BANDS[i]

    cons = solver.Constraint(max(LB, bound - band), min(UB, bound + band))
    #cons = solver.Constraint(LB, UB)

    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

varMaskList = []
for i in range(DAYS):
    varMask = []

    bound = OPT_DAY_BOUNDS[i]
    band = OPT_DAY_BANDS[i]

    from_b = max(LB, bound - band)
    to_b = min(UB, bound + band)

    bound2 = OPT_DAY_BOUNDS[i+1]
    band2 = OPT_DAY_BANDS[i+1]

    from_b2 = max(LB, bound2 - band2)
    to_b2 = min(UB, bound2 + band2)

    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            nday0 = d0 + LB
            nday1 = d1 + LB

            ndaydiff = 0 if i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            penalty = round(penalty, 2)

            vMask = 1

            if penalty > OPT_ACC_COST:
                vMask = 2
            if nday0 < from_b or nday0 > to_b:
                vMask = 0
            elif nday1 < from_b2 or nday1 > to_b2:
                vMask = 0

            varMask.append(vMask)
    varMaskList.append(varMask)

dayMatList = []
for i in range(DAYS):
    dayMat = []

    # constraint on WMAT SUM
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            varMaskFlag = varMaskList[i][d0 * BRANGE + d1]
            var = None
            if varMaskFlag > 0:
                var = solver.IntVar(0, 0 if varMaskFlag == 2 else 1, "dmat_%d_%d_%d" % (i, d0, d1))
            dayMat.append(var)

    cons = solver.Constraint(1, 1)
    any0 = 0
    for var in dayMat:
        if var is not None:
            any0 |= 1
            cons.SetCoefficient(var, 1)
    assert(any0)

    vs = groupedByDay[i]

    # constraints on connection between CHOICE_W AND WMAT
    cons = solver.Constraint(0, 0)
    for (var, n_people) in vs:
        cons.SetCoefficient(var, n_people)

    for ix, var in enumerate(dayMat):
        if var is not None:
            coef = LB + (ix // BRANGE)
            cons.SetCoefficient(var, -coef)
            
            bound = OPT_DAY_BOUNDS[i]
            band = OPT_DAY_BANDS[i]

            from_b = max(LB, bound - band)
            to_b = min(UB, bound + band)
            assert(coef >= from_b and coef <= to_b)

    # constraint on row and next column
    if i > 0:
        prev = dayMatList[-1]
        cur = dayMat

        bound = OPT_DAY_BOUNDS[i]
        band = OPT_DAY_BANDS[i]

        from_b = max(LB, bound - band)
        to_b = min(UB, bound + band)

        for d in range(BRANGE):
            cons = solver.Constraint(0, 0)
            anyMask = 0
            for j in range(BRANGE):
                if prev[BRANGE * j + d] is not None:
                    assert(d + LB >= from_b and d + LB <= to_b)
                    cons.SetCoefficient(prev[BRANGE * j + d], -1)
                    anyMask = anyMask | 1
                if cur[BRANGE * d + j] is not None:
                    assert(d + LB >= from_b and d + LB <= to_b)
                    cons.SetCoefficient(cur[BRANGE * d + j], 1)
                    anyMask = anyMask | 2
            if not (anyMask == 0 or anyMask == 3):
                import pdb; pdb.set_trace()

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
    for d0 in range(BRANGE):
        for d1 in range(BRANGE):
            nday0 = d0 + LB
            nday1 = d1 + LB

            ndaydiff = 0 if i == (DAYS - 1) else abs(nday0 - nday1)

            penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

            var = dayMat[BRANGE * d0 + d1]

            #objective.SetCoefficient(var, penalty)
            penalty = round(penalty, 2)

            if var is None:
                cnt += 1
            else:
                objective.SetCoefficient(var, penalty)
                
print("ignored %d vars" % cnt)

#solver.EnableOutput()
#solver.SetNumThreads(6)
#solver.SetTimeLimit(5 * 24 * 60 * 60 * 1000) # 24h
#solver.SetTimeLimit(12 * 60 * 60 * 1000) # 24h

with open("milp_santa.lp", "w") as fd:
    print(solver.ExportModelAsLpFormat(False), file=fd)

#with open("milp_santa.mps", "w") as fd:
#    print(solver.ExportModelAsMpsFormat(False), file=fd)

os.exit(0)

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print("%s solution found" % ("feasible" if status == solver.FEASIBLE else "optimal"))
    print("solution: " + str(solver.Objective().Value()))

    with open("submission_mip_big_cbc.csv", "w") as fd:
        print("family_id,assigned_day", file=fd)
        for i in range(data.shape[0]):
            lst = [int(familyVars[i][j].solution_value()) for j in range(len(familyVars[i]))]
            print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

else:
    print(status)
    print("no success")
