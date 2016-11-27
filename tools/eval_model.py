__author__ = 'Administrator'
import cfg
from cfg import CFG
import data_reader
import numpy as np
import metric

def evaluate(generator):
    num_test_samples = len(CFG['TRAIN'])
    print 'testing samples:%d'%(num_test_samples)
    resList= {}
    gtList = {}
    IDs = []
    print 'generating sentece...'
    for i in range(num_test_samples):
        _id = CFG['TRAIN'][i]
        vidID, capID = _id.split('_')
        gtList[vidID] = data_reader.data_source['CAPs'][vidID]
        sent = sent_generate(generator, vidID)
        print 'vidID: %s description: %s'%(vidID, sent)
        resList[vidID] = [{u'image_id': vidID, u'caption': sent}]
        IDs.append(vidID)
    print 'evaluating...'
    scorer = metric.COCOScorer()
    eval_res = scorer.score(gtList, resList, IDs)
    return eval_res

def sent_generate(generator, vidID):
    feats = []
    frame_feats = np.copy(data_reader.data_source['FEATs'][vidID])
    feats.append(data_reader.simple_comp_vid_level_feats(frame_feats))
    x_cnn = np.asarray(feats,dtype='float32')

    x_sentence = np.zeros((1, CFG['SEQUENCE LENGTH'] - 1), dtype='int32')
    mask = np.zeros((1,CFG['SEQUENCE LENGTH']),dtype='uint8')
    words = []
    i = 0
    while True:
        mask[0,i] = 1
        p0 = generator(x_cnn, x_sentence, mask)
        pa = p0.argmax(-1)
        tok = pa[0][i]
        word = CFG['idx2word'][tok]
        if word == '<eos>' or i >= CFG['SEQUENCE LENGTH'] - 1:
            return ' '.join(words)
        else:
            x_sentence[0][i] = tok
            words.append(word)
        i += 1

