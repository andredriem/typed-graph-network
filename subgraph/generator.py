import numpy as np
import copy
import itertools
import pycosat
import time


def generate_invalid(base_subgraph):
  while True:

    random_nodes =  np.random.choice([1,2,6,7,8,9,10],[5])
    random_edges = np.random.randint(0,2,[5,5])

    random_nodes[:3] = base_subgraph["nodes"][:3].tolist()

    random_edges[0][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[1][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[2][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[3][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[4][:3] = base_subgraph["edges"][0][:3].tolist()

    # Creates noise on nodes weights or in edges
    if np.random.choice([0,1]):
      random_nodes[np.random.choice([0,1,2])] = np.random.choice([1,2,6,7,8,9,10])
    else:
      pertubation_index_x = np.random.choice([0,1,2])
      pertubation_index_y = np.random.choice([0,1,2])
      random_edges[pertubation_index_x][pertubation_index_y] = \
        (random_edges[pertubation_index_x][pertubation_index_y] + 1) % 2


    x_axis_rotation = np.random.randint(0,5)
    y_axis_rotation = np.random.randint(0,5)

    yield {
      "nodes": np.roll(random_nodes, x_axis_rotation),
      "edges": np.roll( random_edges, [x_axis_rotation, y_axis_rotation], (0,1)),
      }

def generate_valid(base_subgraph):
  while True:

    random_nodes =  np.random.randint(1,11,[5])
    random_edges = np.random.randint(0,2,[5,5])

    random_nodes[:3] = base_subgraph["nodes"][:3].tolist()

    print(random_edges[0][:3])

    random_edges[0][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[1][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[2][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[3][:3] = base_subgraph["edges"][0][:3].tolist()
    random_edges[4][:3] = base_subgraph["edges"][0][:3].tolist()

    x_axis_rotation = np.random.randint(0,5)
    y_axis_rotation = np.random.randint(0,5)

    yield {
      "nodes": np.roll(random_nodes, x_axis_rotation),
      "edges": np.roll( random_edges, [x_axis_rotation, y_axis_rotation], (0,1)),
      }

if __name__ == '__main__':
  nodes_w = np.array([4, 3, 5, 0, 0])
  edges = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ])

  base_subgraph = {'nodes': nodes_w, 'edges': edges }

  generator = generate_invalid(base_subgraph)

  last_time = time.time()
  for batch in generator:
    print("Created batch in {} seconds".format(time.time()-last_time))
    print("Batch: ", batch)
    last_time = time.time()
