__author__ = 'Administrator'
import numpy as np
CFG = {}
CFG['BATCH_SIZE'] = 4
CFG['NUM_EPOCH'] = 30
CFG['SEQUENCE LENGTH'] = 32
CFG['EMBEDDING SIZE'] = 512
CFG['VOCAB SIZE'] = 13010
CFG['VIS SIZE'] = 4096
CFG['DATASET PATH'] = '../youtube2text_iccv15/'
CFG['TRAIN'] = np.load(CFG['DATASET PATH']+'train.pkl')
CFG['TEST'] = np.load(CFG['DATASET PATH'] +'test.pkl')
CFG['VALID'] = np.load(CFG['DATASET PATH'] + 'valid.pkl')
CFG['worddict'] = np.load(CFG['DATASET PATH'] + 'worddict.pkl')

# wordict start with index 2
word_idict = dict()
for kk, vv in CFG['worddict'].iteritems():
    word_idict[vv] = kk
word_idict[0] = '<eos>'
word_idict[1] = 'UNK'
CFG['worddict'] = word_idict