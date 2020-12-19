#Adapted from https://www.kaggle.com/morrisb/how-to-recommend-anything-deep-recommender

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# To create deep learning models
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model


from optparse import OptionParser
import configparser
parser = OptionParser()
parser.add_option('-l','--logname',default='log',help='specify the log name')
parser.add_option('-t','--train',default='reviews_train',help='specify training set')
parser.add_option('-v','--val',default='reviews_test',help='specify validation set')
parser.add_option('-p','--path',default='.',help='specify the path to file input')
parser.add_option('-e','--epoch',type="int",default = 15, help='specific number of epoch')
parser.add_option('-b','--batchsize',type="int",default = 256, help='specific number of batch size')

(options, args) = parser.parse_args()

#path = '/content/drive/My Drive/Colab Notebooks/item/LibraryThings/'
path = options.path

#filename = 'reviews_uir.csv'
filename = str(options.logname) + '.csv'


df_train = pd.read_csv(path+'/'+ options.train + '.csv', sep='\t', names=['User', 'item', 'Rating'], usecols=[0,1,2])
df_test = pd.read_csv(path+'/'+options.val+'.csv', sep='\t', names=['User', 'item', 'Rating'], usecols=[0,1,2])

print('#train = ', len(df_train))
print('#test = ', len(df_test))

# Create user- & item-id mapping
df_merge = [df_train, df_test]
df_filterd = pd.concat(df_merge)

user_id_mapping = {id:i for i, id in enumerate(df_filterd['User'].unique())}
item_id_mapping = {id:i for i, id in enumerate(df_filterd['item'].unique())}

#####################  Matrix Factorization #######################

# Create correctly mapped train- & testset
train_user_data = df_train['User'].map(user_id_mapping)
train_item_data = df_train['item'].map(item_id_mapping)

test_user_data = df_test['User'].map(user_id_mapping)
test_item_data = df_test['item'].map(item_id_mapping)

# Get input variable-sizes
users = len(user_id_mapping)
items = len(item_id_mapping)
ratings = len(df_filterd)

embedding_size = 16
nepochs = options.epoch

print('#users: ', users)
print('#books: ', items)
print('#ratings: ', ratings)

##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

# Create embedding layers for users and items
user_embedding = Embedding(output_dim=embedding_size, 
                           input_dim=users,
                           input_length=1, 
                           name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, 
                            input_dim=items,
                            input_length=1, 
                            name='item_embedding')(item_id_input)

# Reshape the embedding layers
user_vector = Reshape([embedding_size])(user_embedding)
item_vector = Reshape([embedding_size])(item_embedding)

# Compute dot-product of reshaped embedding layers as prediction
y = Dot(1, normalize=False)([user_vector, item_vector])

# Setup model
model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(loss='mse', optimizer='adam')

# Fit model
history = model.fit([train_user_data, train_item_data],
          df_train['Rating'],
          batch_size=options.batchsize, 
          epochs=nepochs,
          validation_split=0.1,
          shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MF - Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test model
y_pred = model.predict([test_user_data, test_item_data])
y_true = df_test['Rating'].values

#  Compute RMSE
rmse_mf = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
mae_mf = mean_absolute_error(y_pred=y_pred, y_true=y_true)
print('RMSE MF:', rmse_mf)
print('MAE  MF:', mae_mf)
fheader = np.array(['#users','#books','ratings','#embedding_size','epochs','rmse_mf','mae_mf'])
fscores = np.array([users, items, ratings, embedding_size, nepochs, rmse_mf, mae_mf    ])
DAT =  np.column_stack((fheader, fscores))
np.savetxt(path+'results/mf-5ratings-' + filename + '-f'+ str(embedding_size) + '-e' + str(nepochs) + '.txt', DAT, delimiter="\t", fmt="%s") 

##################### Deep Matrix Factorization #######################
# Setup variables
user_embedding_size = 16
item_embedding_size = 16
nepochs=2
nodes=128

##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

# Create embedding layers for users and items
user_embedding = Embedding(output_dim=user_embedding_size, 
                           input_dim=users,
                           input_length=1, 
                           name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=item_embedding_size, 
                            input_dim=items,
                            input_length=1, 
                            name='item_embedding')(item_id_input)

# Reshape the embedding layers
user_vector = Reshape([user_embedding_size])(user_embedding)
item_vector = Reshape([item_embedding_size])(item_embedding)

# Concatenate the reshaped embedding layers
concat = Concatenate()([user_vector, item_vector])

# Combine with dense layers
dense = Dense(nodes)(concat)
y = Dense(1)(dense)

# Setup model
model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(loss='mse', optimizer='adam')

# Fit model
history = model.fit([train_user_data, train_item_data],
          df_train['Rating'],
          batch_size=256, 
          epochs=nepochs,
          validation_split=0.1,
          shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('DeepMF - Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test model
y_pred = model.predict([test_user_data, test_item_data])
y_true = df_test['Rating'].values

#  Compute RMSE
rmse_dmf = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
mae_dmf = mean_absolute_error(y_pred=y_pred, y_true=y_true)

print('#users: ', users)
print('#books: ', items)
print('#ratings: ', ratings)
print('RMSE DeepMF:', rmse_dmf)
print('RMSE     MF:', rmse_mf)
print('MAE DeepMF:', mae_dmf)
print('MAE     MF:', mae_mf)

fheader = np.array(['#users','#books','#ratings','#user_embedding_size','book_embedding_size','nodes','epochs','rmse_dmf','mae_dmf'])
fscores = np.array([users, items, ratings, user_embedding_size, item_embedding_size, nodes, nepochs, rmse_dmf, mae_dmf    ])
DAT =  np.column_stack((fheader, fscores))
np.savetxt(path+'results/dmf-5ratings-'+filename+'-fu'+ str(user_embedding_size) + '-fi' + str(item_embedding_size) + '-node' + str(nodes) + '-e' + str(nepochs) + '.txt', DAT, delimiter="\t", fmt="%s") 

