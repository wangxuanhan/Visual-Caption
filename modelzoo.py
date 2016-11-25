__author__ = 'Administrator'
import sys
sys.path.insert(0,'../tools')
import lasagne
from lasagne.layers import InputLayer, EmbeddingLayer,DenseLayer,\
        ConcatLayer, LSTMLayer, DropoutLayer, ReshapeLayer
from lasagne.nonlinearities import softmax
import cfg
from cfg import CFG
import collections

def show_network_configuration(net):

    num_layers = net.__len__()
    layer_names = net.keys()
    print 40*'-'
    print 'network configuration'

    for i in range(num_layers):
        print 'Layer: %s shape:%r'%(layer_names[i],net[layer_names[i]].output_shape)

def build_lstm_decorer():
    net = collections.OrderedDict()
    net['sent_input'] = InputLayer((None, CFG['SEQUENCE LENGTH'] - 1))
    net['word_emb'] = EmbeddingLayer(net['sent_input'], input_size=CFG['VOCAB SIZE'],\
                                    output_size=CFG['EMBEDDING SIZE'])
    net['vis_input'] = InputLayer((None, CFG['VIS SIZE']))
    net['vis_emb'] = DenseLayer(net['vis_input'], num_units=CFG['EMBEDDING SIZE'],
                                nonlinearity=lasagne.nonlinearities.identity)
    net['vis_emb_reshp'] = ReshapeLayer(net['vis_emb'],(-1,1,CFG['EMBEDDING SIZE']))
    net['decorder_input'] = ConcatLayer([net['vis_emb_reshp'], net['word_emb']])
    net['feat_dropout'] = DropoutLayer(net['decorder_input'],p=0.5)
    net['lstm'] = LSTMLayer(net['feat_dropout'],num_units=CFG['EMBEDDING SIZE'],
                            grad_clipping=5.)
    net['lstm_dropout'] = DropoutLayer(net['lstm'], p=0.5)
    net['lstm_reshp'] = ReshapeLayer(net['lstm_dropout'], (-1,CFG['EMBEDDING SIZE']))
    net['word_prob'] = DenseLayer(net['lstm_reshp'], num_units=CFG['VOCAB SIZE'],
                                  nonlinearity=softmax)
    net['sent_prob'] = ReshapeLayer(net['word_prob'],(-1,CFG['SEQUENCE LENGTH'], CFG['VOCAB SIZE']))

    return net

if __name__ == '__main__':
    show_network_configuration(build_lstm_decorer())