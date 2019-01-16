import numpy as np
from utils import get_word_form
from utils import is_vb_in

class Features():
    def __init__(
            self,
            words_list,
            pos_list,
            heads_list,
            features_to_include_list):

        # p-word, p-pos
        self.f1 = set()
        # p-word
        self.f2 = set()
        # p-pos
        self.f3 = set()
        # c-word, c-pos
        self.f4 = set()
        # c-word
        self.f5 = set()
        # c-pos
        self.f6 = set()
        # p-word,c-word,p-pos,c-pos
        self.f7 = set()
        # c-word, c-pos, p-pos
        self.f8 = set()
        # c-word, p-word, c-pos
        self.f9 = set()
        # p-word, c-pos, p-pos
        self.f10 = set()
        # c-word, p-word, p-pos
        self.f11 = set()
        # c-word, p-word
        self.f12 = set()
        # c-pos, p-pos
        self.f13 = set()
        # p-form, c-form
        self.f14 = set()
        # p-pos, c-pos, is-vb-between
        self.f15 = set()
        # p-word, p-pos, c-form
        self.f16 = set()
        # c-word, c-pos, p-form
        self.f17 = set()
        # p-pos, c-pos, in-pos
        self.f18 = set()
        # p-pos, c-pos, p[-1]-pos, c[-1]-pos
        self.f19 = set()
        # p-pos, c-pos, p[+1]-pos, c[+1]-pos
        self.f20 = set()
        # p-pos, c-pos, c[+1]-pos, c[+2]-pos
        self.f21 = set()
        # p-pos, c-pos, c[-1]-pos, c[-2]-pos
        self.f22 = set()
        # p-pos, c-pos, p[-1]-pos
        self.f23 = set()
        # p-pos, c-pos, c[-1]-pos
        self.f24 = set()
        # p-pos, c-pos, c[+1]-pos
        self.f25 = set()
        # p-pos, c-pos, p[+1]-pos
        self.f26 = set()
        # p-pos, c-pos, c[-1]-pos, c[+1]-pos
        self.f27 = set()
        # p-pos, c-pos, p[-1]-pos, p[+1]-pos
        self.f28 = set()

        # fill for each feature it's valus according to train
        self.create_all_features_lists(words_list, pos_list, heads_list)
        print('Finished lists init')
        # list of feature lists each feature i is located in index i - 1
        feature_list = [self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7,self.f8,self.f9,
                        self.f10,self.f11,self.f12,self.f13,self.f14,self.f15,self.f16,self.f17,
                        self.f18, self.f19, self.f20, self.f21, self.f22, self.f23, self.f24, self.f25,
                        self.f26, self.f27, self.f28]
        # list of feature configurations for each feature i its configuration is located in index i - 1
        feature_requirements_list = [('p-w','p-p'),('p-w',),('p-p',),('c-w','c-p'),('c-w',),('c-p',),
                                     ('p-w','c-w','p-p','c-p'),('c-w','p-p','c-p'),('p-w','c-w','c-p'),
                                     ('p-w', 'p-p', 'c-p'),('p-w','c-w','p-p'),('p-w','c-w'),('p-p','c-p'),
                                     ('p-f','c-f'),('p-p','c-p','vb-in'),('p-w','p-p','c-f'),('c-w','p-w','p-f'),
                                     ('all-p-in'), ('p-p','c-p','p[-1]-pos','c[-1]-pos'),('p-p','c-p','p[+1]-pos','c[+1]-pos'),
                                     ('p-p', 'c-p','c[+1]-pos','c[+2]-pos'),('p-p', 'c-p','c[-1]-pos','c[-2]-pos'),
                                     ('p-p', 'c-p','p[-1]-pos'),('p-p', 'c-p','c[-1]-pos'),
                                     ('p-p', 'c-p', 'p[+1]-pos'), ('p-p', 'c-p', 'c[+1]-pos'),
                                     ('p-p', 'c-p', 'c[-1]-pos', 'c[+1]-pos'),('p-p', 'c-p', 'p[-1]-pos', 'p[+1]-pos')]

        self.feature_wieghts_len = 0
        self.actual_feature_dict = {}
        if features_to_include_list == 'ALL':
            features_to_include_list = list(range(1,len(feature_list) + 1))
        # creates indexes to only features that participate in the model
        for included_feature in features_to_include_list:
            self.actual_feature_dict[included_feature] = {}
            # creates mapping for each feature value to it's location in feature weights vector
            self.actual_feature_dict[included_feature]['idx_mapping'] = self.create_feature_to_idx_mapping_dict(
                            list(feature_list[included_feature - 1]),
                            self.feature_wieghts_len)
            self.actual_feature_dict[included_feature]['requrements'] = feature_requirements_list[included_feature - 1]
            self.feature_wieghts_len += len(feature_list[included_feature - 1])
            print ("Num Of Features F" + str(included_feature) + ' : ' + str(len(feature_list[included_feature - 1])))

    def create_all_features_lists(
            self,
            word_list,
            pos_list,
            head_list):
        # initiating all feature lists to features that participates in train
        curr_sent_words = []
        curr_sent_pos   = []
        curr_sent_heads = []
        for i in range(len(word_list)):
            if ((word_list[i] == 'ROOT') and (i != 0)) or (i == len(word_list) - 1):
                if (i == len(word_list) - 1) and (word_list[i] != 'ROOT'):
                    curr_sent_words.append(word_list[i])
                    curr_sent_pos.append(pos_list[i])
                    curr_sent_heads.append(head_list[i])
                for j in range(1, len(curr_sent_words)):
                    # create vars for features
                    curr_head_idx = int(curr_sent_heads[j])
                    m_word = curr_sent_words[j]
                    h_word = curr_sent_words[curr_head_idx]
                    m_pos  = curr_sent_pos[j]
                    h_pos  = curr_sent_pos[curr_head_idx]
                    curr_sent_len = len(curr_sent_words) - 1
                    h_form = get_word_form(h_word)
                    m_form = get_word_form(m_word)
                    is_vb_between = is_vb_in(j,curr_head_idx,curr_sent_pos)
                    all_possible_dist = list(range(1, len(curr_sent_words))) + list(range(-1*(len(curr_sent_words) -1), 0))
                    # insert vars to correct features
                    for t in all_possible_dist:
                        curr_y = t
                        self.f3.add((h_pos, curr_y))
                        self.f6.add((m_pos, curr_y))
                        self.f2.add((h_word, curr_y))
                        self.f5.add((m_word, curr_y))
                        self.f1.add((h_word, h_pos, curr_y))
                        self.f4.add((m_word, m_pos, curr_y))
                        self.f13.add((h_pos, m_pos, curr_y))
                        self.f10.add((h_word, h_pos, m_pos, curr_y))
                        self.f8.add((m_word, h_pos, m_pos, curr_y))
                        self.f14.add((h_form, m_form, curr_y))
                        self.f15.add((h_pos, m_pos, is_vb_between, curr_y))
                        self.f16.add((h_word, h_pos, m_form, curr_y))
                        self.f17.add((m_word, m_pos, h_form, curr_y))

                    curr_y = j - curr_head_idx
                    if j > curr_head_idx:
                        inner_sent_pos = curr_sent_pos[curr_head_idx + 1: j]
                    else:
                        inner_sent_pos = curr_sent_pos[j + 1:curr_head_idx]
                    for in_pos in inner_sent_pos:
                        self.f18.add((h_pos, m_pos, in_pos, curr_y))
                    if curr_head_idx != 0:
                        self.f19.add((h_pos, m_pos, curr_sent_pos[curr_head_idx - 1], curr_sent_pos[j-1], curr_y))
                        self.f23.add((h_pos, m_pos, curr_sent_pos[curr_head_idx - 1], curr_y))
                    if max(j, curr_head_idx) < (curr_sent_len ):
                        self.f20.add((h_pos, m_pos, curr_sent_pos[curr_head_idx + 1], curr_sent_pos[j + 1], curr_y))
                    if j < curr_sent_len - 1:
                        self.f21.add((h_pos, m_pos, curr_sent_pos[j + 1], curr_sent_pos[j + 2], curr_y))
                        self.f25.add((h_pos, m_pos, curr_sent_pos[j + 1], curr_y))
                        self.f27.add((h_pos, m_pos, curr_sent_pos[j - 1], curr_sent_pos[j + 1], curr_y))
                    if j > 1:
                        self.f22.add((h_pos, m_pos, curr_sent_pos[j - 1], curr_sent_pos[j - 2], curr_y))
                        self.f24.add((h_pos, m_pos, curr_sent_pos[j - 1], curr_y))
                    if curr_head_idx < curr_sent_len:
                        self.f26.add((h_pos, m_pos, curr_sent_pos[curr_head_idx + 1], curr_y))
                        if curr_head_idx != 0:
                            self.f28.add((h_pos, m_pos, curr_sent_pos[curr_head_idx - 1], curr_sent_pos[curr_head_idx + 1], curr_y))

                    self.f7.add((h_word, m_word, h_pos, m_pos, curr_y))
                    self.f11.add((h_word, m_word, h_pos, curr_y))
                    self.f9.add((h_word, m_word, m_pos, curr_y))
                    self.f12.add((h_word, m_word, curr_y))

                curr_sent_words = ['ROOT']
                curr_sent_pos   = ['ROOT']
                curr_sent_heads = ['ROOT']
            else:
                curr_sent_words.append(word_list[i])
                curr_sent_pos.append(pos_list[i])
                curr_sent_heads.append(head_list[i])


    def create_feature_to_idx_mapping_dict(
            self,
            all_vals,
            start_idx):
        # for all feature vals creates mapping to weight vec index
        mapping_dict = {}
        for i in range(len(all_vals)):
            mapping_dict[all_vals[i]] = start_idx + i
        return mapping_dict

    def get_feature_indexs(
            self,
            head_word,
            head_pos,
            curr_word,
            curr_pos,
            head_idx,
            curr_idx,
            vb_in,
            all_sent_pos,
            print_err = False):
        # for current word and potential head returns all it's feature indexes
        curr_y = curr_idx - head_idx
        feature_idx_list = []
        sent_last_idx = len(all_sent_pos) - 1
        for feature_kind in self.actual_feature_dict:
            feat_vars = []
            for req in self.actual_feature_dict[feature_kind]['requrements']:
                if req == 'c-w':
                    feat_vars.append(curr_word)
                elif req == 'p-w':
                    feat_vars.append(head_word)
                elif req == 'c-p':
                    feat_vars.append(curr_pos)
                elif req == 'p-p':
                    feat_vars.append(head_pos)
                elif req == 'c-f':
                    feat_vars.append(get_word_form(curr_word))
                elif req == 'p-f':
                    feat_vars.append(get_word_form(head_word))
                elif req == 'vb-in':
                    feat_vars.append(vb_in)
                elif req == 'c[-1]-pos':
                    if curr_idx > 0:
                        feat_vars.append(all_sent_pos[curr_idx - 1])
                elif req == 'c[+1]-pos':
                    if curr_idx < sent_last_idx:
                        feat_vars.append(all_sent_pos[curr_idx + 1])
                elif req == 'c[-2]-pos':
                    if curr_idx > 1:
                        feat_vars.append(all_sent_pos[curr_idx - 2])
                elif req == 'c[+2]-pos':
                    if curr_idx < sent_last_idx - 1:
                        feat_vars.append(all_sent_pos[curr_idx + 2])
                elif req == 'p[-1]-pos':
                    if head_idx > 0:
                        feat_vars.append(all_sent_pos[head_idx - 1])
                elif req == 'p[+1]-pos':
                    if head_idx < sent_last_idx:
                        feat_vars.append(all_sent_pos[head_idx + 1])

            if feature_kind not in [2,3,5,6]:
                feat_vars.append(curr_y)
                feat_vars = tuple(feat_vars)
            elif feature_kind == 18:
                for t in range(min(head_idx, curr_idx) + 1, max(head_idx, curr_idx)):
                    feat_vars = (head_pos, curr_pos, all_sent_pos[t], curr_y)
                    try:
                        feature_idx_list.append(self.actual_feature_dict[feature_kind]['idx_mapping'][feat_vars])
                    except Exception as e:
                        pass
            else:
                feat_vars = (feat_vars[0], curr_y)
            try:
                feature_idx_list.append(self.actual_feature_dict[feature_kind]['idx_mapping'][feat_vars])
            except Exception as e:
                if print_err == True:
                    pass

        return feature_idx_list