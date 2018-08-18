import tensorflow as tf 

"""
Definition of RNN Cell
"""
def rnn_cell(is_train,num_neurons):
  with tf.variable_scope("cell_def",reuse = tf.AUTO_REUSE):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons)
    
    if is_train:
      cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=0.5)
    return cell
    

"""
Definition of sbot class - Char RNN class
"""
class sbot:
  
  def __init__(self,input_shape,num_neurons,num_layers,beta1,temperature,is_train):
    self.batch_size = input_shape[0]
    self.seq_length = input_shape[1]
    self.char_size = input_shape[2]
    self.lr_rate = tf.placeholder(tf.float32,shape=[])   #learning rate taken at runtime
    self.beta1 = beta1
    self.is_train = is_train                             #boolean variable to hold the mode (training or inference)
    self.num_neurons = num_neurons
    self.num_layers = num_layers
    self.temperature = temperature
    self.inputs,self.targets = self.model_io()
    self.logits, self.pred, self.fin_state = self.model_arc(self.inputs,self.is_train,self.num_neurons,self.num_layers,self.temperature)
    self.loss = self.model_loss(self.logits,self.targets)
    self.opt = self.model_opt(self.loss,self.lr_rate,self.beta1)
  
  """
  Function to model the input and output of the network.
  """
  def model_io(self):
    inputs = tf.placeholder(dtype=tf.float32,shape=(self.batch_size,self.seq_length,self.char_size))
    outputs = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.seq_length,self.char_size))
    return inputs, outputs
  
  """
  Model Architecture Definition 
  """
  def model_arc(self,inputs,is_train,num_neurons,num_layers,temperature):
    
    with tf.variable_scope("sbot", reuse = False if is_train==True else True):
      
      multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(is_train,num_neurons) for _ in range(num_layers)])
      self.init_state = multi_cell.zero_state(batch_size = self.batch_size,dtype = tf.float32)
      
      with tf.variable_scope("rnn_def",reuse = tf.AUTO_REUSE):
        rnn_out,state = tf.nn.dynamic_rnn(multi_cell,inputs,initial_state = self.init_state,time_major=False)
      
      rnn_out = tf.concat(rnn_out,axis=1)
      rnn_out = tf.reshape(rnn_out,[-1,num_neurons])
      with tf.variable_scope('fc_layer',reuse = tf.AUTO_REUSE):
        w = tf.Variable(initial_value=tf.truncated_normal(shape=[num_neurons, self.char_size], stddev=0.1))
        b = tf.Variable(tf.zeros(self.char_size))
      logits = tf.nn.xw_plus_b(rnn_out, w, b)
      logits = logits/temperature
      pred = tf.nn.softmax(logits)
      pred = tf.reshape(pred,[-1,self.seq_length,self.char_size])
      return logits,pred,state
  
  """
  Function to calculate loss.
  """
  def model_loss(self,logit,label):
    tf.stop_gradient(label)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=label))
    #tf.summary.histogram("loss",loss)
    return loss
  
  """
  Optimization using Adam Optimizer.
  """
  def model_opt(self,loss,lr_rate,beta1):
    train_vars = tf.trainable_variables()
    sbot_vars = [var for var in train_vars if var.name.startswith('sbot')]     
        
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        sbot_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(loss,var_list = sbot_vars)
        return sbot_opt
    
