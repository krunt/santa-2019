
import pandas as pd
from ortools.graph import pywrapgraph

#min_cost_flow = pywrapgraph.SimpleMinCostFlow()
min_cost_flow = pywrapgraph.SimpleMaxFlow()

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

MAX_CAP = 300
DAY_OFFS = 101
START_NODE = 0
END_NODE = 1 + 100 + data.shape[0]

for i in range(data.shape[0]):
    (day0, day1) = data.iloc[i][["choice_0", "choice_1"]]
    family_node_id = DAY_OFFS + i

    n_people = int(data.iloc[i]["n_people"])

    #min_cost_flow.AddArcWithCapacityAndUnitCost(day0, family_node_id, n_people, 0)
    #min_cost_flow.AddArcWithCapacityAndUnitCost(day1, family_node_id, n_people, 50)

    min_cost_flow.AddArcWithCapacity(day0, family_node_id, n_people)
    min_cost_flow.AddArcWithCapacity(day1, family_node_id, n_people)

for i in range(1,101):
    #min_cost_flow.AddArcWithCapacityAndUnitCost(START_NODE, i, MAX_CAP, 0)
    min_cost_flow.AddArcWithCapacity(START_NODE, i, MAX_CAP)

for i in range(data.shape[0]):
    family_node_id = DAY_OFFS + i
    n_people = int(data.iloc[i]["n_people"])
    #min_cost_flow.AddArcWithCapacityAndUnitCost(family_node_id, END_NODE, n_people, 0)
    min_cost_flow.AddArcWithCapacity(family_node_id, END_NODE, n_people)

if min_cost_flow.Solve(START_NODE, END_NODE) == min_cost_flow.OPTIMAL:
    print('Max flow:', min_cost_flow.OptimalFlow())
    print('')
    print('  Arc    Flow / Capacity')
    for i in range(min_cost_flow.NumArcs()):
      print('%1s -> %1s   %3s  / %3s' % (
          min_cost_flow.Tail(i),
          min_cost_flow.Head(i),
          min_cost_flow.Flow(i),
          min_cost_flow.Capacity(i)))
    print('Source side min-cut:', min_cost_flow.GetSourceSideMinCut())
    print('Sink side min-cut:', min_cost_flow.GetSinkSideMinCut())


#for i in range(END_NODE + 1):
#    min_cost_flow.SetNodeSupply(i, 0)



#if min_cost_flow.SolveMaxFlowWithMinCost() == min_cost_flow.OPTIMAL:
#    print('Minimum cost:', min_cost_flow.OptimalCost())
#    print('Maximum flow:', min_cost_flow.MaximumFlow())
#    print('Num Arcs:', min_cost_flow.NumArcs())
#    print('')
#    print('  Arc    Flow / Capacity  Cost')
#    for i in range(min_cost_flow.NumArcs()):
#        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
#        print('%1s -> %1s   %3s  / %3s       %3s' % (
#            min_cost_flow.Tail(i),
#            min_cost_flow.Head(i),
#            min_cost_flow.Flow(i),
#            min_cost_flow.Capacity(i),
#            cost))
#else:
#  print('There was an issue with the min cost flow input.')
