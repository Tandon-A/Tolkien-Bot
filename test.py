import tensorflow as tf 
import numpy as np 
import pickle
import os 

"""
Import sbot class definition.
"""
from sbot import sbot 

"""
Function to pick a character based on the output probabilities of the model. 
"""
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

"""
Test function.
"""
def test(net,length,ind_to_char,char_size,model_dir):
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,model_dir)
    print ("\n")
    new_state = sess.run(net.init_state)
    init_char = np.random.randint(char_size)
    sample = ind_to_char[init_char] 
    print (ind_to_char[init_char],end="")
    for i in range(length):
      input_seq = np.zeros((1,1,char_size))
      input_seq[0,0,init_char] = 1
      pred,new_state = sess.run([net.pred,net.fin_state],feed_dict={net.inputs:input_seq,net.init_state:new_state})
      #init_char = np.argmax(pred[0,0,:])
      init_char = weighted_pick(pred[0,0,:])
      print (ind_to_char[init_char], end="")
      sample = sample + ind_to_char[init_char]
    return sample
    

def main(_): 
    if not os.path.exists(FLAGS.meta_dir + "tr_file_meta"):
        print ("Training meta data file doesn't exist.")
    else:
        if not os.path.exists(FLAGS.model_dir):
          print ("Model file doesn't exist")
          
        else:
        
          file = open(FLAGS.meta_dir+"tr_file_meta","rb")
          f2d = pickle.load(file)
          char_size = f2d[0]
          ind_to_char = f2d[1]
          char_to_ind = f2d[2]
          file.close()

          seq_length = 1
          batch_size = 1
          input_shape = batch_size,seq_length,char_size
          beta1= 0.5 
          num_layers = 4
          num_neurons = 512
          is_train = False #testing mode
          length = 500
          temperature = 1.0 
          tf.reset_default_graph()
          net = sbot(input_shape,num_neurons,num_layers,beta1,temperature,is_train)
          sample = test(net,length,ind_to_char,char_size,FLAGS.model_dir)
          print ("\n\n\nsampling finished")


flags = tf.app.flags
flags.DEFINE_string("meta_dir",None,"Directory where training file metadata is stored")
flags.DEFINE_string("model_dir",None,"Directory where model files are stored")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
    
