# This file is a implementation of a GNN to solve the subgraph problem.

import sys, os
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tgn import TGN
from mlp import Mlp


def build_network(d):
  # Define hyperparameters
  d = d
  learning_rate = 2e-5
  l2norm_scaling = 1e-10
  global_norm_gradient_clipping_ratio = 0.65


  # Placeholder for answers to the decision problems (one per problem)
  subgraph_exists = tf.placeholder( tf.float32, shape = (None,), name = 'subgraph_exists' )
  # Placeholders for the list of number of vertices per instance
  n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
  # Placeholder for the adjacency matrix connecting each edge to its source and target vertices
  VV_matrix   = tf.placeholder( tf.float32, shape = (None,None), name = "VV" )
  # Placeholder for the column matrix of edge weights
  vertice_weight = tf.placeholder( tf.float32, shape = (None,1), name = "vertice_weight" )
  # Placeholder for the number of timesteps the GNN is to run for
  time_steps = tf.placeholder( tf.int32, shape = (), name = "time_steps" )

  # All edges embeddings are initialized with the same value, which is a trained parameter learned by the network
  total_n = tf.shape(VV_matrix)[1]
  v_init = tf.get_variable(initializer=tf.random_normal((1,d)), dtype=tf.float32, name='V_init')
  vertex_initial_embeddings = tf.tile(
      tf.div(v_init, tf.sqrt(tf.cast(d, tf.float32))),
      [total_n, 1]
  )


 # Define GNN dictionary
  GNN = {}

  # Configure GNN
  gnn = TGN(
        {
            'V': d,
        },
        {
            'VV': ('V','V')
        },
        {
            'V_msg_V': ('V','V'),
        },
        {
            'V': [
                {
                    'mat': 'VV',
                    'msg': 'V_msg_V',
                    'var': 'V'
                }
            ],
        },
        name='SUBGRAPH'
    )

  # Populate GNN dictionary
  GNN['gnn']          = gnn
  GNN['subgraph_exists'] = subgraph_exists
  GNN['n_vertices']   = n_vertices
  GNN['VV']           = VV_matrix
  GNN['W']            = vertice_weight
  GNN['time_steps']   = time_steps

  # Define V_vote, which will compute one logit for each vertice
  # <--- André: The network witch asks each node if it thinks its part of the subgraph right?
  V_vote_MLP = Mlp(
      layer_sizes = [ d for _ in range(3) ],
      activations = [ tf.nn.relu for _ in range(3) ],
      output_size = 1,
      name = 'E_vote',
      name_internal_layers = True,
      kernel_initializer = tf.contrib.layers.xavier_initializer(),
      bias_initializer = tf.zeros_initializer()
      )

  # Get the last embeddings
  last_states = gnn(
    { "VV": VV_matrix, 'W': vertice_weight},
    { "V": vertex_initial_embeddings},
    time_steps = time_steps
  )
  GNN["last_states"] = last_states
  V_n = last_states['V'].h

  # Compute a vote for each embedding
  #E_vote = tf.reshape(E_vote_MLP( tf.concat([E_n,target_cost],axis=1) ), [-1])
  V_vote = tf.reshape(V_vote_MLP(V_n), [-1])

  # Compute the number of problems in the batch
  num_problems = tf.shape(n_vertices)[0]

  # Compute a logit probability for each problem <- I'll look into this
  pred_logits = tf.while_loop(
      lambda i, pred_logits: tf.less(i, num_problems),
      lambda i, pred_logits:
          (
              (i+1),
              pred_logits.write(
                  i,
                  tf.reduce_mean(V_vote[tf.reduce_sum(n_vertices[0:i]):tf.reduce_sum(n_vertices[0:i])+n_vertices[i]])
              )
          ),
      [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
      )[1].stack()
  # Convert logits into probabilities
  GNN['predictions'] = tf.sigmoid(pred_logits)

  # Compute True Positives, False Positives, True Negatives, False Negatives, accuracy
  GNN['TP'] = tf.reduce_sum(tf.multiply(subgraph_exists, tf.cast(tf.equal(subgraph_exists, tf.round(GNN['predictions'])), tf.float32)))
  GNN['FP'] = tf.reduce_sum(tf.multiply(subgraph_exists, tf.cast(tf.not_equal(subgraph_exists, tf.round(GNN['predictions'])), tf.float32)))
  GNN['TN'] = tf.reduce_sum(tf.multiply(tf.ones_like(subgraph_exists)-subgraph_exists, tf.cast(tf.equal(subgraph_exists, tf.round(GNN['predictions'])), tf.float32)))
  GNN['FN'] = tf.reduce_sum(tf.multiply(tf.ones_like(subgraph_exists)-subgraph_exists, tf.cast(tf.not_equal(subgraph_exists, tf.round(GNN['predictions'])), tf.float32)))
  GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(subgraph_exists, tf.round(GNN['predictions'])), tf.float32))

  # Define loss
  GNN['loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=subgraph_exists, logits=pred_logits))

  # Define optimizer
  optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

  # Compute cost relative to L2 normalization
  vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])

  # Define gradients and train step
  grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
  GNN['train_step'] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))

  # Return GNN dictionary
  return GNN


if __name__ == "__main__":
  build_network(10)
