__author__ = 'Xuanhan Wang'
import _init_paths
import model_factory
import eval_model
import data_reader as data
from cfg import CFG
import numpy as np
import cPickle
import time
import lasagne
#-------------prepare models---------------------------
def exp_model():
    if CFG['USE_MODEL'] == 1:
        print 'using pretrained vid caption model:%s'%(CFG['MODEL PATH'])
        paramfile = open(CFG['MODEL PATH'],'r')
        net_params = cPickle.load(paramfile)
        paramfile.close()
        model = model_factory.build_vidcaption_model()
        lasagne.layers.set_all_param_values(model[model['net name']]['word_prob'],net_params)
    else:
        print 'build new vidcaption model'
        model = model_factory.build_vidcaption_model()
    model_factory.show_network_configuration(model[model['net name']])
    print 'network computation graph completed!!!'
    print 40 * '*'
    print 'start compiling neccessary experimental functions'
    print 'compiling...'
    exp_func = model_factory.train_test_func(model)

    return model,exp_func

def do_train_exp():
    print 'Training mp_lstm Network'
    print 'Experimental settins:'
    print 'NUMBER EPOCH: %d'%(CFG['NUM_EPOCH'])
    print 'BATCH SIZE: %d'%(CFG['BATCH_SIZE'])
    print 40*'-'

    print 40*'-'
    print 'build model...'
    model,exp_func = exp_model()
    print 'model ok'
    print 40*'*'
    num_samples = len(CFG['TRAIN'])
    num_batch = num_samples / CFG['BATCH_SIZE']
    best_loss = np.inf
    acc=[]
    print 'training...'
 #   eval_model.evaluate(exp_func['sent prob'])
    for iepoch in np.arange(CFG['NUM_EPOCH']):

        epoch_loss = 0.
        epoch_acc =0.
        for ibatch in np.arange(num_batch):
            batch_idx = CFG['TRAIN'][ibatch*CFG['BATCH_SIZE']:(ibatch+1)*CFG['BATCH_SIZE']]
            batch_data,batch_words,mask = data.get_batch_data(batch_idx)
            predict_words = np.reshape(batch_words,(-1,))
            predict_words = predict_words[np.where(mask.flatten()==1)]
            print batch_data.shape
            print batch_words.shape
            print mask.shape
            print predict_words.shape
#            word2sent(batch_words)
            print 'forward and backward...'
            batch_loss,batch_acc= exp_func['train func'](batch_data,batch_words,mask,predict_words)

            print '%d epoch %d batch: loss %f acc %f'%(iepoch+1, ibatch+1, batch_loss, batch_acc)
            epoch_loss += batch_loss
            epoch_acc += batch_acc
        epoch_loss /= num_batch
        epoch_acc /= num_batch
        train_acc = epoch_acc
        logfile = open('logs/VIDCAP_MP/log_msvd_mplstm_'+time.strftime('%Y-%m-%d',time.localtime(time.time())),'a+')
        print >> logfile,time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(time.time()))+'\n script train loss:%f train acc:%f'%(epoch_loss, train_acc)
        logfile.close()
        print 'mean batch loss: %f'%epoch_loss
        print 'mean batch acc: %f'%epoch_acc
        print 40*'-'
        if epoch_loss < best_loss:
            print 'find better training result.'
            print 'saving model'
            net_params = lasagne.layers.get_all_param_values(model[model['net name']]['word_prob'])
            modelfile = open('../models/VIDCAP_MP/msvd_mplstm_params.pkl','wb')
            cPickle.dump(net_params,modelfile)
            modelfile.close()
            if iepoch>=2:
                print 'lets validating our model'
                eval_model.evaluate(exp_func['sent prob'])
    for i in np.arange(len(acc)):
        print '%d epoch acc %f'%(i+1,acc[i])


def word2sent(wordids):
    print 'Captions:'
    for item in wordids:
        cap = ''
        for i in item:
            cap = cap+CFG['idx2word'][i]+' '
        print cap
    return
if __name__ == '__main__':
    do_train_exp()
