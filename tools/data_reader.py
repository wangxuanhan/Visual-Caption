__author__ = 'Administrator'
import cfg
from cfg import CFG
import numpy as np

data_source = {}
data_source['FEATs'] = np.load(CFG['DATASET PATH']+'FEAT_key_vidID_value_features.pkl')
data_source['CAPs'] = np.load(CFG['DATASET PATH'] + 'CAP.pkl')

def get_words(vidID, capID):
        caps = data_source['CAPs'][vidID]
        rval = None
        for cap in caps:
            if cap['cap_id'] == capID:
                rval = cap['tokenized'].split(' ')
                break
        assert rval is not None
        return rval

def simple_comp_vid_level_feats(frame_feats,pool_function=np.mean,axis=0):
    return pool_function(frame_feats,axis=axis).flatten()

def get_batch_data(batch_idx):
    feats = []
    words = []
    for item in batch_idx:
        vidID, capID = item.split('_')
        frame_feats = np.copy(data_source['FEATs'][vidID])
        feats.append(simple_comp_vid_level_feats(frame_feats))
        sentence = get_words(vidID, capID)
        words.append([CFG['worddict'][w]
                     if CFG['worddict'][w] < CFG['VOCAB SIZE'] else 1 for w in sentence])

    return np.asarray(feats,dtype='float32'),np.asarray(words,dtype='int32')