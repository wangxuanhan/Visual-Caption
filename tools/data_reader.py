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

def rnn_input_reorganized(sents):

    masks = []
    seqs = []
    for s in sents:
        mask = np.zeros((CFG['SEQUENCE LENGTH'],),dtype='uint8')
        mask[0] = 1  # visual feature must be inputted
        words = np.zeros((CFG['SEQUENCE LENGTH'],),dtype='int32')
        if len(s) < CFG['SEQUENCE LENGTH']-1:
            words[:len(s)] = s[:]
            mask[1:len(s)+1] = 1
        else:
            words[:-1] = s[:CFG['SEQUENCE LENGTH']-1]
            mask[:] = 1

        masks.append(mask)
        seqs.append(words)

    return np.asarray(seqs,dtype='int32'), np.asarray(masks,dtype='uint8')


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
    words, masks = rnn_input_reorganized(words)
    return np.asarray(feats,dtype='float32'), words, masks

def test_fn():
    batch_idx = CFG['TRAIN'][:CFG['BATCH_SIZE']]
    feats,words,masks = get_batch_data(batch_idx)
    print feats.shape
    print words.shape
    print masks.shape

if __name__ == '__main__':
    test_fn()