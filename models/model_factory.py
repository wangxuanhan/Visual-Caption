from modelzoo import build_lstm_decorer
import theano
from theano import tensor as T
import lasagne

import collections

def show_network_configuration(net):

    num_layers = net.__len__()
    layer_names = net.keys()
    print 40*'-'
    print 'network configuration'

    for i in range(num_layers):
        print 'Layer: %s shape:%r'%(layer_names[i],net[layer_names[i]].output_shape)

def classifier_train_test_config(net,model):
    net_params = lasagne.layers.get_all_params(net['word_prob'],trainable=True)
    mask = lasagne.layers.get_output(net['mask_input'])
    fmask = T.flatten(mask)
    targets = T.ivector()
    # training configuration
    train_prob = lasagne.layers.get_output(net['word_prob'])
    train_prob = train_prob[T.nonzero(fmask)]
    train_class_loss = lasagne.objectives.categorical_crossentropy(train_prob,targets)
    train_loss = T.mean(train_class_loss)
    train_pred = T.argmax(train_prob,axis=1)
    train_acc = T.mean(T.eq(train_pred,targets))
    # Testing configuration
    sent_prob = lasagne.layers.get_output(net['sent_prob'],deterministic=True)

    model['ground truth'] = targets
    model['train loss'] = train_loss
    model['train accuracy'] = train_acc
    model['sent prob'] = sent_prob
    model['params'] = net_params
    return model

def train_test_func(MODEL,optimizer=1,learning_rate = 0.005):
    exp_funcs = {}
    if optimizer == 0:
        print 'Optimizer: general sgd '
        updates = lasagne.updates.sgd(MODEL['train loss'], MODEL['params'][:], learning_rate)
    else:
        print 'Optimizer: adadelta'
        updates = lasagne.updates.adadelta(MODEL['train loss'], MODEL['params'][:], learning_rate)

    print 'General settings: Learning rate(%f)  Momentum(%f)  NumberFunctions(%d) NumberParams(%d)'%\
          (learning_rate,0.9,2,MODEL['params'].__len__())

    updates = lasagne.updates.apply_momentum(updates,MODEL['params'][:])

    net_name = MODEL['net name']
    exp_funcs['train func'] = theano.function(inputs=[MODEL[net_name]['vis_input'].input_var,
                                                    MODEL[net_name]['sent_input'].input_var,
                                                    MODEL[net_name]['mask_input'].input_var,
                                                    MODEL['ground truth']],
                               outputs=[MODEL['train loss'],MODEL['train accuracy']],
                               updates=updates)
    exp_funcs['sent prob']= theano.function(inputs=[MODEL[net_name]['vis_input'].input_var,
                                                    MODEL[net_name]['sent_input'].input_var,
                                                    MODEL[net_name]['mask_input'].input_var],
                              outputs=MODEL['sent prob'])

    return exp_funcs


def build_vidcaption_model(net_name='mp_lstm'):
    net = build_lstm_decorer()

    model = collections.OrderedDict()
    model['net name'] = net_name
    model[net_name] = net
    model = classifier_train_test_config(net,model)

    return model
