import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import copy
import itertools
import pycosat
import time


def generate_invalid(base_subgraph):
  while True:    
    base_subgraph_nx = nx.from_numpy_matrix(base_subgraph, create_using=nx.MultiGraph())
    invalid_subgraph_candidate = np.random.randint(0,2,[5,5])
    invalid_subgraph_candidate_nx = nx.from_numpy_matrix(invalid_subgraph_candidate, create_using=nx.MultiGraph())
    valid_subgraph = True
    
    while valid_subgraph:
      valid_subgraph = isomorphism.MultiGraphMatcher(invalid_subgraph_candidate_nx,base_subgraph_nx).subgraph_is_isomorphic()
      if valid_subgraph:        
        base_subgraph_nx = nx.from_numpy_matrix(base_subgraph, create_using=nx.MultiGraph())
        invalid_subgraph_candidate = np.random.randint(0,2,[5,5])
        invalid_subgraph_candidate_nx = nx.from_numpy_matrix(invalid_subgraph_candidate, create_using=nx.MultiGraph())
        print(invalid_subgraph_candidate)

    yield invalid_subgraph_candidate

def generate_valid(base_subgraph):
  while True:

    random_edges = np.random.randint(0,2,[5,5])

    random_edges[0][:3] = base_subgraph[0][:3].tolist()
    random_edges[1][:3] = base_subgraph[0][:3].tolist()
    random_edges[2][:3] = base_subgraph[0][:3].tolist()
    random_edges[3][:3] = base_subgraph[0][:3].tolist()
    random_edges[4][:3] = base_subgraph[0][:3].tolist()

    x_axis_rotation = np.random.randint(0,5)
    y_axis_rotation = np.random.randint(0,5)

    yield np.roll( random_edges, [x_axis_rotation, y_axis_rotation], (0,1))

if __name__ == '__main__':
  edges = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ])

  generator = generate_valid(edges)

  last_time = time.time()
  for batch in generator:
    print("Created batch in {} seconds".format(time.time()-last_time))
    print("Batch: ", batch)
    last_time = time.time()
