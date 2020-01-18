
import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph

min_cost_flow = pywrapgraph.SimpleMinCostFlow()

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

MAX_CAP = 240
DAY_OFFS = 101
START_NODE = 0
END_NODE = 1 + 100 + data.shape[0]

for i in range(data.shape[0]):
    (day0, day1) = data.iloc[i][["choice_0", "choice_1"]]
    family_node_id = DAY_OFFS + i

    n_people = int(data.iloc[i]["n_people"])

    min_cost_flow.AddArcWithCapacityAndUnitCost(day0, family_node_id, n_people, 0)
    min_cost_flow.AddArcWithCapacityAndUnitCost(day1, family_node_id, n_people, round(50 / n_people))

for i in range(1,101):
    min_cost_flow.AddArcWithCapacityAndUnitCost(START_NODE, i, MAX_CAP, 0)

for i in range(data.shape[0]):
    family_node_id = DAY_OFFS + i
    n_people = int(data.iloc[i]["n_people"])
    min_cost_flow.AddArcWithCapacityAndUnitCost(family_node_id, END_NODE, n_people, 0)

for i in range(END_NODE + 1):
    min_cost_flow.SetNodeSupply(i, 0)

total_people = 0
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])
    total_people += n_people
#print("total_people=" + str(total_people))

min_cost_flow.SetNodeSupply(START_NODE, total_people)
min_cost_flow.SetNodeSupply(END_NODE, -total_people)

if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
#    print('Minimum cost:', min_cost_flow.OptimalCost())
#    print('Maximum flow:', min_cost_flow.MaximumFlow())
#    print('Num Arcs:', min_cost_flow.NumArcs())
#    print('')
#    print('  Arc    Flow / Capacity  Cost')

    arr = np.zeros((data.shape[0], 2))
    for i in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Flow(i) == 0:
            continue
        if min_cost_flow.Tail(i) >= 1 and min_cost_flow.Tail(i) <= 100 and min_cost_flow.Head(i) != END_NODE:
            family_id = min_cost_flow.Head(i) - DAY_OFFS
            arr[family_id, 0] = family_id
            arr[family_id, 1] = min_cost_flow.Tail(i)

    print("family_id,assigned_day")
    for i in range(arr.shape[0]):
        print("%d,%d" % (i, int(arr[i, 1])))

#    for i in range(min_cost_flow.NumArcs()):
#        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
#        print('%1s -> %1s   %3s  / %3s       %3s' % (
#            min_cost_flow.Tail(i),
#            min_cost_flow.Head(i),
#            min_cost_flow.Flow(i),
#            min_cost_flow.Capacity(i),
#            cost))
else:
  print('There was an issue with the min cost flow input.')
