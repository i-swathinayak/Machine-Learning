import numpy as np

from util import accuracy
from hmm import HMM


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################

    state_dict = {}
    obs_dict = {}
    tag_index = 0
    word_index = 0

    for sentence in train_data:
        for tag in sentence.tags:
            if tag not in state_dict:
                state_dict[tag] = tag_index
                tag_index += 1
        for word in sentence.words:
            if word not in obs_dict:
                obs_dict[word] = word_index
                word_index += 1

    L = len(train_data)
    obs_count = len(obs_dict)
    state_count = len(state_dict)

    pi = np.zeros(state_count)
    for sentence in train_data:
        pi[state_dict.get(sentence.tags[0])] += 1

    pi = pi / L
    A = np.zeros([state_count, state_count])
    B = np.zeros([state_count, obs_count])

    tag_count = np.ones(state_count)
    tag_trans_count = np.ones(state_count)
    
    for sentence in train_data:
        c = len(sentence.tags)
        for i in range(c):
            tag_count[state_dict.get(sentence.tags[i])] += 1
            if i != c - 1:
                tag_trans_count[state_dict.get(sentence.tags[i])] += 1

    for sentence in train_data:
        for c in range(len(sentence.tags)):
            B[state_dict.get(sentence.tags[c]), obs_dict.get(sentence.words[c])] += 1

    for i in range(state_count):
        B[i, :] = B[i, :] / tag_count[i]

    for sentence in train_data:
        for c in range(len(sentence.tags) - 1):
            A[state_dict.get(sentence.tags[c]), state_dict.get(sentence.tags[c + 1])] += 1

    for i in range(state_count):
        A[i, :] = A[i, :] / tag_trans_count[i]

    model = HMM(pi, A, B, obs_dict, state_dict)

    ###################################################
    return model


# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
  
    for sentence in test_data:
        osequence=np.array(sentence.words)
        path=model.viterbi(osequence)
        tagging.append(path)    
    
    ###################################################
    return tagging