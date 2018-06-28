import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import math

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

#set directory paths
'''
root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

'''
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print("training set")
#print(train)
sample_submission = pd.read_csv("gender_submission.csv")
train.head()


#print("validation - y")
#print(val_y)
#print("train - x")
#print(len(train_x))
#print("train")
#print(train)
train_x_sex = train.Sex.values[:]
#print("train_x_sex")
#print(train_x_sex)


#train and test data cleanup or manipulation

for i in range(len(train_x_sex)):
	if train_x_sex[i] == 'male':
		train_x_sex[i] = 0
	elif train_x_sex[i] == 'female':
		train_x_sex[i] = 1
	else:
		train_x_sex[i] = -1

train.Sex.values[:] = train_x_sex

#the averate age of known people with known age is 30 (precisely 29.699)
for i in range(train.Age.values.size):
	if np.isnan (train.Age.values[i]):
		train.Age.values[i] = 30.0

for i in range(len(train.Embarked.values[:])):
	#print(x)
	if train.Embarked.values[i] == 'C':
		train.Embarked.values[i] = 67
		#print(train.Embarked.values[i])
	elif train.Embarked.values[i] == 'Q':
		train.Embarked.values[i] = 81
	elif train.Embarked.values[i] == 'S':
		train.Embarked.values[i] = 83
	else:
		train.Embarked.values[i] = 67
	#train.Embarked.values[i] = ord(x)
	#print(train.Embarked.values[i])

#print(train)
#train_x = train.dropna(subset=['Survived'])
train_without_y = train.drop('Survived', 1)

#divide the 
split_size = int(train.shape[0]*0.7)
#print ("split size", split_size)

train_x, val_x = train_without_y[:split_size], train_without_y[split_size:]
train_y, val_y = train.Survived.values[:split_size], train.Survived.values[split_size:]

#print(val_x)


#HELPER Functions below
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)    
    p = p.reindex(batch_x=batch_x)
    #p[batch_x] = p[batch_x].astype(int)
    
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

	
### set all variables
train_x_mat = train_x.as_matrix()

print("train_x.size", train_x.shape)
print("train_x_mat.size", train_x_mat.shape[1])
#print(train_x)
# number of neurons in each layer
input_num_units = train_x_mat.shape[1]	#to be reviewed
hidden_num_units = 10	#to be reviewed
output_num_units = 2	#to be reviewed


# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 1
batch_size = train_x_mat.shape[0]
learning_rate = 0.01

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer() # tf.initialize_all_variables()


with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print ("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))
    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
	
	
print("---- End of the code ----")
