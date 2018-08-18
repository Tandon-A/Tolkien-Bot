import tensorflow as tf 
import numpy as np 
import math
import pickle
import os

"""
Import sbot class definition.
"""
from sbot import sbot  

"""
Function to load data and prepare the metadata variables
"""
def load_data(file_path,meta_dir):
  with open(file_path,"r") as f:
    data = f.read()
  f.close()

  chars = list(set(data))
  char_size = len(chars)
  ind_to_char = {ind:char for ind,char in enumerate(chars)}
  char_to_ind = {char:ind for ind,char in enumerate(chars)}

  file = open(meta_dir + "tr_file_meta","wb")
  d2f = []
  d2f.append(char_size)
  d2f.append(ind_to_char)
  d2f.append(char_to_ind)
  pickle.dump(d2f,file)
  file.close()  
  return data,char_size,char_to_ind
  

"""
Function to build training sequences.
"""
def get_data(data,char_to_ind,seq_length,char_size):
  x = []
  y = []
  for i in range(int(len(data)/seq_length)):
    x_seq =  data[i*seq_length:(i+1)*seq_length]
    x_seq_ind = [char_to_ind[val] for val in x_seq]
    x_seq = np.zeros((seq_length,char_size))
    y_seq =  data[((i*seq_length) + 1):((i+1)*seq_length) + 1]
    y_seq_ind = [char_to_ind[val] for val in y_seq]
    y_seq = np.zeros((seq_length,char_size))
    for j in range(seq_length):
      x_seq[j][x_seq_ind[j]] = 1
      y_seq[j][y_seq_ind[j]] = 1
    x.append(x_seq)
    y.append(y_seq)
  return np.array(x),np.array(y)


"""
Training Function.
"""
def train(net,tr_data,tr_labels,batch_size,max_iter,lr_rate,model_dir):
  saver = tf.train.Saver(max_to_keep=None)
  epoch = 0
  print ("starting training")
  max_len = tr_data.shape[0] - batch_size
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while epoch < max_iter:
      bs = 0
      print (epoch)
      if epoch >= 25 and epoch%5 ==0:
        lr_rate = lr_rate * 0.97;      #decreasing learning rate after every 5 epochs, starting 25th epoch. 
      while bs < max_len:
        batch_data = tr_data[bs:(bs+batch_size)]
        batch_labels = tr_labels[bs:(bs+batch_size)]
       
        sess.run(net.opt,feed_dict={net.inputs:batch_data,net.targets:batch_labels,net.lr_rate:lr_rate}) 
        
        if epoch % 5 == 0  and bs % 5000 == 0:
          cost = sess.run(net.loss,feed_dict={net.inputs:batch_data,net.targets:batch_labels,net.lr_rate:lr_rate})
          print ("epoch = %r step = %r cost = %r" %(epoch,bs,cost))
        
        bs = bs + batch_size
        counter = counter + 1 
        train_writer.add_summary(summary, counter)
      
      epoch = epoch + 1        
    
    saver.save(sess,model_dir,write_meta_graph=True)
    print ("### Model Saved ### epoch = %r" %(epoch))
   
   
def main(_): 
    if not os.path.exists(FLAGS.file_path):
        print ("Training text file doesn't exist.")
    else:
            
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        if not os.path.exists(FLAGS.meta_dir):
            os.makedirs(FLAGS.meta_dir)
        
        seq_length = 100
        data,char_size,char_to_ind = load_data(FLAGS.file_path,FLAGS.meta_dir)
        
        tr_data,tr_label = get_data(data,char_to_ind,seq_length,char_size)
        print (tr_data.shape,tr_label.shape)
        
        batch_size = 50
        input_shape = batch_size,seq_length,char_size
        lr_rate = 0.002
        beta1= 0.5 
        num_layers = 4
        num_neurons = 512
        is_train = True
        max_iter = 50
        temperature = 1.0
        
        tf.reset_default_graph()
        net = sbot(input_shape,num_neurons,num_layers,beta1,temperature,is_train)
        train(net,tr_data,tr_label,batch_size,max_iter,lr_rate,FLAGS.model_dir)
    


flags = tf.app.flags
flags.DEFINE_string("file_path",None,"Path to file containing training texts")
flags.DEFINE_string("meta_dir","meta_dir/","Directory to store training file metadata produced by application")
flags.DEFINE_string("model_dir","sbot_model/","Directory name to save checkpoints")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
