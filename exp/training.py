"""
Experiment for training arbitry model on dataset with Tensorflow1.X
"""
import argparse
import random
import sys
import time

import numpy as np
import tensorflow as tf
import wandb
from keras import backend
from sklearn.utils import shuffle
from tqdm import tqdm

sys.path.append('/code/adaptive-hessian-free-optimization/')
from src.optimizer.hfoptimizer_adam import AdaptiveHessianFreeOptimizer
from src.data.load_dataset import load_data
from src.models import build_model

def reset_graph(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

##########################################################################################
# watch the lower bound of L-conjugate smoothness.
# L ≧ 2( f(w_t) - f(w_t-1) - <g(w_t), w_t - w_t-1> ) / ||w_t - w_t-1||^{2}
##########################################################################################
def calc_cg_L_smooth(sess, neighbor_params, cg_delta_values, batch_loesses):
    params_delta = [next_param - current_param for next_param, current_param in zip(neighbor_params[-1], neighbor_params[0])]
    
    cg_params_delta_dots = sess.run(
        tf.reduce_sum(
            [tf.matmul(a=tf.reshape(cg, shape=[-1, 1]), b=tf.reshape(param_delta, shape=[-1, 1]), transpose_a=True) 
            for cg, param_delta 
            in zip(cg_delta_values, params_delta)
            ]
        )
    )
    
    params_delta_euclidnorm = sess.run(tf.reduce_sum([tf.norm(param_delta, ord=2) for param_delta in params_delta]))                    
    
    return 2 * (batch_loesses[-1] - batch_loesses[-2] - cg_params_delta_dots) / params_delta_euclidnorm

def eval_model(sess, loss, accuracy, feed_dict):
    
    loss_eval = sess.run(loss, feed_dict=feed_dict)
    acc_eval  = sess.run(accuracy, feed_dict=feed_dict)
    
    return loss_eval, acc_eval

def main():
    ###############################################################################
    # load data and serup model
    ###############################################################################
    dataloader = load_data()
    train_X, test_X, train_y, test_y = load_data().load_cifar10()

    n_batches  = train_X.shape[0] // args.batch_size
    batch_loesses = []

    reset_graph(args.seed)
    x, t, is_training, y_out, loss, accuracy = build_model(model_name=args.arch)

    ###############################################################################
    # training & eval model
    ###############################################################################
    with tf.Session(config=config) as sess:
        backend.set_session(sess)
        ### Initializing optimizer ###
        hf_optimizer = AdaptiveHessianFreeOptimizer(
            sess=sess,
            loss=loss,
            output=y_out,
            learning_rate=args.lr,
            damping=args.damping,
            batch_size=args.batch_size,
            use_gauss_newton_matrix=False,
            beta2 = args.beta2,
            cg_epsilon = args.cg_epsilon,
            cg_sigma = args.cg_sigma,
            cg_L_smoothness = args.cg_L_smoothness,
            dtype=tf.float32
        )
        hf_optimizer.info()
        
        init = tf.global_variables_initializer()
        init.run()
        for epoch in range(args.epochs):
            _train_X, _train_y = shuffle(train_X, train_y, random_state=epoch)
            
            neighbor_params = [] # to watch L-conjugate-smoothness

            for iteration in tqdm(range(n_batches)):
                start   = iteration * args.batch_size
                end     = start + args.batch_size
                x_batch = _train_X[start:end]
                t_batch = _train_y[start:end]

                cg_delta_tensors, lr, lr_numerator, lr_denominator = hf_optimizer.minimize({x: x_batch, t: t_batch, is_training: True})
                # cg_delta_values = sess.run(cg_delta_tensors, feed_dict={x: x_batch, t: t_batch, is_training: False})
                # cg_norm = sess.run(tf.reduce_sum([tf.norm(delta, ord=1) for delta in cg_delta_values]))

                batch_loss = sess.run(loss, feed_dict={x: x_batch, t: t_batch, is_training: False})
                batch_loesses.append(batch_loss)
                
                params = sess.run(tf.trainable_variables(), feed_dict={x: x_batch, t: t_batch, is_training: False})
                neighbor_params.append(params)
                if len(neighbor_params) > 2:
                    del neighbor_params[0]

                # if iteration != 0:                      
                if (iteration % 10 == 0) or (iteration == n_batches-1):

                    # cg_L_smoothness = calc_cg_L_smooth(sess=sess, neighbor_params=neighbor_params, cg_delta_values=cg_delta_values, batch_loesses=batch_loesses)
                    train_loss, train_acc = eval_model(sess=sess, loss=loss, accuracy=accuracy, feed_dict={x: train_X[:10000], t: train_y[:10000], is_training: False})  
                    test_loss, test_acc   = eval_model(sess=sess, loss=loss, accuracy=accuracy, feed_dict={x: test_X, t: test_y, is_training: False})
                    
                    wandb.log({
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'test_loss': test_loss,
                        'test_accuracy': test_acc,
                        # 'cg_norm': cg_norm,
                        # 'cg_L smooth': np.abs(cg_L_smoothness),
                        'lr': lr,
                        'lr_numerator': lr_numerator,
                        'lr_denominator': lr_denominator,
                    })
    
if __name__ == "__main__":

    ###############################################################################
    # Training settings
    ###############################################################################
    parser = argparse.ArgumentParser(description='Adaptive Hessian-free optimizer bentchmark on CIFAR-10')

    parser.add_argument('--gpu-num', type = str, default = '1', metavar = 'G', help = 'GPU Device number')
    parser.add_argument('--seed', type = int, default = 1, metavar = 'S', help = 'random seed for training')
    parser.add_argument('--batch-size', type = int, default = 128, metavar = 'B', help = 'input batch size for training')
    parser.add_argument('--epochs', type = int, default = 10, metavar = 'E', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.001, metavar = 'L', help = 'learning rate')
    parser.add_argument('--damping', type = float, default = 10, metavar = 'D', help = 'damping')
    parser.add_argument('--beta2', type = float, default = 1e-8, metavar = 'B2', help = 'hyper param of Adam')
    
    parser.add_argument('--cg-epsilon', type = float, default = 1e-3)
    parser.add_argument('--cg-sigma', type = int, default = 50)
    parser.add_argument('--cg-L-smoothness', type = int, default = 100)
    
    parser.add_argument('--arch', type = str, default = 'LeNet', metavar = 'A', help = 'model architecture')

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    ###############################################################################
    # gpu config setting
    ###############################################################################
    config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu_num, # specify GPU number
        allow_growth=True
        )
    )

    ###############################################################################
    # wandb setup
    ###############################################################################
    hyperparams = {
        'batch_size': args.batch_size,
        'epochs':args.epochs,
        'learning_rate':args.lr,
        'damping':args.damping,
        'beta_2':args.beta2,
        'model_architecture':args.arch,
        'cg_epsilon': args.cg_epsilon,
        'cg_sigma': args.cg_sigma,
        'cg_L_smoothness': args.cg_L_smoothness,
    }
    exp_name = time.strftime('%Y%m%d-%H%M%S') + 'hf-adam'
    wandb.init(config=hyperparams, project=f'cifar10-{args.arch}-hf-experiment', entity='mtkwt', name=exp_name)

    main()