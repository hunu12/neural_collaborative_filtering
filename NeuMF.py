'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, multiply, Reshape, Flatten, Dropout, Concatenate, CategoryEncoding
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model, evaluate_per_interactionLevel
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--meta_info', type=int, default=0,
                        help='Whether to use meta data information of user and item')
    return parser.parse_args()


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, lenUserInfo=0, lenItemInfo=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    user_info = Input(shape=(lenUserInfo,), dtype='float32', name='user_info')
    item_info = Input(shape=(lenItemInfo,), dtype='float32', name='item_info')
    
    if lenUserInfo > 0:
        user_oneHot = Flatten()(CategoryEncoding(num_tokens=num_users, output_mode="one_hot")(user_input))
        mf_user_latent = Dense(mf_dim, name='mf_embedding_user', kernel_initializer=initializers.RandomNormal(stddev=0.01), 
                               kernel_regularizer=l2(reg_mf))(Concatenate(axis=1)([user_oneHot, user_info]))
        mlp_user_latent = Dense(int(layers[0]/2), name='mlp_embedding_user', kernel_initializer=initializers.RandomNormal(stddev=0.01), 
                                kernel_regularizer=l2(reg_layers[0]))(Concatenate(axis=1)([user_oneHot, user_info]))
    else:
        MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = initializers.RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_mf), input_length=1)
        MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                  embeddings_initializer = initializers.RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
        
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))

    if lenItemInfo > 0:
        item_oneHot = Flatten()(CategoryEncoding(num_tokens=num_items, output_mode="one_hot")(item_input))
        mf_item_latent = Dense(mf_dim, name='mf_embedding_item', kernel_initializer=initializers.RandomNormal(stddev=0.01), 
                               kernel_regularizer=l2(reg_mf))(Concatenate(axis=1)([item_oneHot, item_info]))
        mlp_item_latent = Dense(int(layers[0]/2), name='mlp_embedding_item', kernel_initializer=initializers.RandomNormal(stddev=0.01), 
                               kernel_regularizer=l2(reg_layers[0]))(Concatenate(axis=1)([item_oneHot, item_info]))                      
    else:
        MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = initializers.RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_mf), input_length=1)   
        MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                  embeddings_initializer = initializers.RandomNormal(stddev=0.01), embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # MF part
    mf_vector = multiply([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_vector = Concatenate(axis=1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = Concatenate(axis=1)([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input, user_info, item_info], 
                  outputs=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset, meta_info=args.meta_info)
    train, testRatings, testNegatives, trainInteractionLevel = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, dataset.trainInteractionLevel
    userInfo, itemInfo = dataset.userInfo, dataset.itemInfo
    lenUserInfo = userInfo.shape[1] if args.meta_info else 0
    lenItemInfo = itemInfo.shape[1] if args.meta_info else 0
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, lenUserInfo=lenUserInfo, lenItemInfo=lenItemInfo)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users,num_items,mf_dim, lenUserInfo=lenUserInfo, lenItemInfo=lenItemInfo)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers, lenUserInfo=lenUserInfo, lenItemInfo=lenItemInfo)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
        
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, userInfo=userInfo, itemInfo=itemInfo)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    hr_ndcg_per_il = evaluate_per_interactionLevel(hits, ndcgs, interactionLevel)
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    for name, hr_per_il, ndcg_per_il in hr_ndcg_per_il.items:
        print(f'\t{name:4s} HR = {hr_per_il:.4f}, NDCG = {ndcg_per_il:.4f}')

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True) 
        
    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Training
        inputs = [np.array(user_input), np.array(item_input)]
        inputs.append(userInfo[user_input] if args.meta_info else np.empty((len(user_input),0)))
        inputs.append(itemInfo[item_input] if args.meta_info else np.empty((len(user_input),0)))
        hist = model.fit(inputs, #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, userInfo=userInfo, itemInfo=itemInfo)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            hr_ndcg_per_il = evaluate_per_interactionLevel(hits, ndcgs, interactionLevel)
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            for name, hr_per_il, ndcg_per_il in hr_ndcg_per_il.items:
                print(f'\t{name:4s} HR = {hr_per_il:.4f}, NDCG = {ndcg_per_il:.4f}')
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))
