import numpy as np
import networkx as nx
import pylab as plt

points_list = [(0,4), (4,3), (4,5), (3,1), (3,2), (3,4), (2,3), (1,5)]

G = nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos,  node_color='b',node_size=500)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=23, font_color='red')
plt.show()

MAX_SIZE = len(G)
print(MAX_SIZE)
goal = 5
R = np.matrix(np.ones(shape=(MAX_SIZE, MAX_SIZE), dtype=np.int64))
R *= -1
Q = np.matrix(np.zeros( (MAX_SIZE, MAX_SIZE), dtype=np.float64 ))
gamma = 0.8

for point in points_list:
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0
    if point[0] == 7:
        R[point[::-1]] = 100
    else:
         R[point[::-1]] = 0
R[goal, goal] = 100


#train
def take_action(state):
    possibles_next_action = np.where(R[state,:] >= 0)[1]
    if possibles_next_action.shape[0] > 1:
        action = int(np.random.choice( possibles_next_action , size=1))
    else:
        action = int(possibles_next_action)
    return action

def update(state, action):
    #new_state = action
    max_index_range = np.where(Q[action,:] == np.max(Q[action,:]))[1]

    if max_index_range.shape[0] > 1:
        max_index = int(np.random.choice(max_index_range, size=1))
    else:
         max_index = int(max_index_range)
    max_value = Q[action, max_index]
    Q[state, action] = R[state, action] + gamma * max_value

    if np.max(Q) > 0:
        return (np.sum(Q/np.max(Q) * 100))
    return 0

state = 0
scores = []
Q = np.matrix(np.zeros( (MAX_SIZE, MAX_SIZE), dtype=np.float64 ))
for i in range(700):
    state_num = np.random.randint(0, Q.shape[0])
    action = take_action(state_num)
    #print(score, end=' ')
    score = update(state_num, action)
    scores.append(score)
    #print (str(score), end=' ')

print("Trained Q matrix:")
print(Q/np.max(Q)*100)
