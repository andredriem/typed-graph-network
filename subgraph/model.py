import sys, os
if __name__ == "__main__":
  sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf
import numpy as np
# Import model builder
from tgn import TGN
from mlp import Mlp
from cnf import CNF

def build_network(d):

  # Hyperparameters
  learning_rate = 2e-5
  parameter_l2norm_scaling = 1e-10
  global_norm_gradient_clipping_ratio = 0.65

  # Define placeholder for satisfiability statuses (one per problem)
  instance_is_graph = tf.placeholder( tf.float32, [ None ], name = "instance_is_graph" )
  time_steps = tf.placeholder(tf.int32, shape=(), name='time_steps')
  matrix_placeholder = tf.placeholder( tf.float32, [ None, None ], name = "VV" )
  num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )
  n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')

  # Compute number of problems
  p = tf.shape( instance_is_graph )[0]

  # All edges embeddings are initialized with the same value, which is a trained parameter learned by the network
  total_n = tf.shape(matrix_placeholder)[1]
  v_init = tf.get_variable(initializer=tf.random_normal((1,d)), dtype=tf.float32, name='V_init')
  vertex_initial_embeddings = tf.tile(
      tf.div(v_init, tf.sqrt(tf.cast(d, tf.float32))),
      [total_n, 1]
  )

  # Define Typed Graph Network
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

  # Define V_vote
  V_vote_MLP = Mlp(
    layer_sizes = [ d for _ in range(3) ],
    activations = [ tf.nn.relu for _ in range(3) ],
    output_size = 1,
    name = "V_vote",
    name_internal_layers = True,
    kernel_initializer = tf.contrib.layers.xavier_initializer(),
    bias_initializer = tf.zeros_initializer()
  ) 

  # Get the last embeddings
  V_n = gnn(
    { "VV": matrix_placeholder },
    {"V": vertex_initial_embeddings},
    time_steps
  )["V"].h
  V_vote = V_vote_MLP( V_n )

  # Reorganize votes' result to obtain a prediction for each problem instance
  def _vote_while_cond(i, p, n_acc, n, n_var_list,predicted_is_graph, L_vote):
    return tf.less( i, p )
  #end _vote_while_cond
      
  predicted_is_graph= tf.TensorArray( size = p, dtype = tf.float32 )
  predicted_is_graph = tf.while_loop(
      lambda i, pred_logits: tf.less(i, p),
      lambda i, pred_logits:
          (
              (i+1),
              pred_logits.write(
                  i,
                  tf.reduce_mean(V_vote[tf.reduce_sum(n_vertices[0:i]):tf.reduce_sum(n_vertices[0:i])+n_vertices[i]])
              )
          ),
      [0, tf.TensorArray(size=p, dtype=tf.float32)]
      )
      
  predicted_is_graph= predicted_is_graph[1].stack()
 
  # Define loss, accuracy
  predict_costs = tf.nn.sigmoid_cross_entropy_with_logits( labels = instance_is_graph, logits =predicted_is_graph)
  predict_cost = tf.reduce_mean( predict_costs )
  vars_cost = tf.zeros([])
  tvars = tf.trainable_variables()
  for var in tvars:
    vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
  #end for
  loss = tf.add( predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
  optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
  grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
  train_step = optimizer.apply_gradients( zip( grads, tvars ) )
  
  accuracy = tf.reduce_mean(
    tf.cast(
      tf.equal(
        tf.cast( instance_is_graph, tf.bool ),
        tf.cast( tf.round( tf.nn.sigmoid(predicted_is_graph) ), tf.bool )
      )
      , tf.float32
    )
  )

  # Define neurosat dictionary
  neurosat = {}
  neurosat["M"]                    = matrix_placeholder
  neurosat["time_steps"]           = time_steps
  neurosat["gnn"]                  = gnn
  neurosat["instance_is_graph"]    = instance_is_graph
  neurosat["predicted_is_graph"]   = predicted_is_graph
  neurosat["num_vars_on_instance"] = num_vars_on_instance
  neurosat["loss"]                 = loss
  neurosat["accuracy"]             = accuracy
  neurosat["train_step"]           = train_step

  return neurosat
#end build_neurosat

if __name__ == "__main__":
  build_network(5)