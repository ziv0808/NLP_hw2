import numpy as np

class Featues():
    def __init__(
            self,
            words_list,
            pos_list,
            heads_list,
            features_to_include_list):

        # create indexes for all features by the features participating in model
        # init all feature lists
        self.create_all_features_lists(words_list, pos_list, heads_list)
        print('Finished lists init')
        # p-word, p-pos
        self.f1 = self.h_word_pos_pair_list
        # p-word
        self.f2 = self.h_word_feat_list
        # p-pos
        self.f3 = self.h_pos_feat_list
        # c-word, c-pos
        self.f4 = self.m_word_pos_pair_list
        # c-word
        self.f5 = self.m_word_feat_list
        # c-pos
        self.f6 = self.m_pos_feat_list
        # p-word,c-word,p-pos,c-pos
        self.f7 = self.two_words_two_pos_list
        # c-word, c-pos, p-pos
        self.f8 = self.m_word_two_pos_list
        # c-word, p-word, c-pos
        self.f9 = self.m_pos_two_words_list
        # p-word, c-pos, p-pos
        self.f10 = self.h_word_two_pos_list
        # c-word, p-word, p-pos
        self.f11 = self.h_pos_two_words_list
        # c-word, p-word
        self.f12 = self.word_pairs_list
        # c-pos, p-pos
        self.f13 = self.pos_pair_list

        feature_list = [self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7,self.f8,self.f9,
                        self.f10,self.f11,self.f12,self.f13]
        feature_requirements_list = [('p-w','p-p'),('p-w',),('p-p',),('c-w','c-p'),('c-w',),('c-p',),
                                     ('p-w','c-w','p-p','c-p'),('c-w','p-p','c-p'),('p-w','c-w','c-p'),
                                     ('p-w', 'p-p', 'c-p'),('p-w','c-w','p-p'),('p-w','c-w'),('p-p','c-p')]

        self.feature_wieghts_len = 0
        self.actual_feature_dict = {}
        if features_to_include_list == 'ALL':
            features_to_include_list = list(range(1,len(feature_list) + 1))
        # creates indexes to only features that participate in the model
        for included_feature in features_to_include_list:
            self.actual_feature_dict[included_feature] = {}
            self.actual_feature_dict[included_feature]['idx_mapping'] = self.create_feature_to_idx_mapping_dict(
                            feature_list[included_feature - 1],
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
        h_pos_feat_list        = []
        m_pos_feat_list        = []
        h_word_feat_list       = []
        m_word_feat_list       = []
        h_word_pos_pair_list   = []
        m_word_pos_pair_list   = []
        pos_pair_list          = []
        word_pairs_list        = []
        h_word_two_pos_list    = []
        m_word_two_pos_list    = []
        h_pos_two_words_list   = []
        m_pos_two_words_list   = []
        two_words_two_pos_list = []

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
                    curr_head_idx = int(curr_sent_heads[j])
                    m_word = curr_sent_words[j]
                    h_word = curr_sent_words[curr_head_idx]
                    m_pos  = curr_sent_pos[j]
                    h_pos  = curr_sent_pos[curr_head_idx]
                    curr_y = (j,curr_head_idx)
                    h_pos_feat_list.append((h_pos, curr_y))
                    m_pos_feat_list.append((m_pos, curr_y))
                    h_word_feat_list.append((h_word, curr_y))
                    m_word_feat_list.append((m_word, curr_y))
                    h_word_pos_pair_list.append((h_word, h_pos, curr_y))
                    m_word_pos_pair_list.append((m_word, m_pos, curr_y))
                    pos_pair_list.append((h_pos, m_pos, curr_y))
                    word_pairs_list.append((h_word, m_word, curr_y))
                    h_word_two_pos_list.append((h_word, h_pos, m_pos, curr_y))
                    m_word_two_pos_list.append((m_word, h_pos, m_pos, curr_y))
                    h_pos_two_words_list.append((h_word, m_word, h_pos, curr_y))
                    m_pos_two_words_list.append((h_word, m_word, m_pos, curr_y))
                    two_words_two_pos_list.append((h_word, m_word, h_pos, m_pos, curr_y))

                curr_sent_words = ['ROOT']
                curr_sent_pos   = ['ROOT']
                curr_sent_heads = ['ROOT']
            else:
                curr_sent_words.append(word_list[i])
                curr_sent_pos.append(pos_list[i])
                curr_sent_heads.append(head_list[i])

        self.h_pos_feat_list = list(set(h_pos_feat_list))
        self.m_pos_feat_list = list(set(m_pos_feat_list))
        self.h_word_feat_list = list(set(h_word_feat_list))
        self.m_word_feat_list = list(set(m_word_feat_list))
        self.h_word_pos_pair_list = list(set(h_word_pos_pair_list))
        self.m_word_pos_pair_list = list(set(m_word_pos_pair_list))
        self.pos_pair_list = list(set(pos_pair_list))
        self.word_pairs_list = list(set(word_pairs_list))
        self.h_word_two_pos_list = list(set(h_word_two_pos_list))
        self.m_word_two_pos_list = list(set(m_word_two_pos_list))
        self.h_pos_two_words_list = list(set(h_pos_two_words_list))
        self.m_pos_two_words_list = list(set(m_pos_two_words_list))
        self.two_words_two_pos_list = list(set(two_words_two_pos_list))

        return

    # def create_all_word_pos_combinations_lists(
    #         self,
    #         word_list,
    #         pos_list):
    #
    #     h_pos_feat_list         = []
    #     m_pos_feat_list         = []
    #     h_word_feat_list        = []
    #     m_word_feat_list        = []
    #     word_pos_pair_list      = []
    #     pos_pair_list           = []
    #     word_pairs_list         = []
    #     word_two_pos_list       = []
    #     pos_two_words_list      = []
    #     two_words_two_pos_list  = []
    #
    #     curr_sent_words = []
    #     curr_sent_pos   = []
    #     for i in range(len(word_list)):
    #         if ((word_list[i] == 'ROOT') and (i != 0)) or (i == len(word_list) -1):
    #             if (i == len(word_list) -1) and (word_list[i] != 'ROOT'):
    #                 curr_sent_words.append(word_list[i])
    #                 curr_sent_pos.append(pos_list[i])
    #             for j in range(len(curr_sent_words)):
    #                 for t in range(len(curr_sent_words)):
    #                     if j != t:
    #                         h_pos_feat_list.append((curr_sent_pos[t]))
    #                         m_pos_feat_list.append((curr_sent_pos[j]))
    #                         h_word_feat_list.append((curr_sent_words[t]))
    #                         m_word_feat_list.append((curr_sent_words[j]))
    #                         word_pos_pair_list.extend([(curr_sent_words[j], curr_sent_pos[j]), (curr_sent_words[t], curr_sent_pos[t])])
    #                         pos_pair_list.extend([(curr_sent_pos[j], curr_sent_pos[t]),(curr_sent_pos[j], curr_sent_pos[t])])
    #                         word_pairs_list.append((curr_sent_words[j], curr_sent_words[t]))
    #                         word_two_pos_list.extend([(curr_sent_words[j], curr_sent_pos[j],curr_sent_pos[t]),(curr_sent_words[j], curr_sent_pos[t],curr_sent_pos[j])])
    #                         pos_two_words_list.extend([(curr_sent_words[j], curr_sent_words[t],curr_sent_pos[j]),(curr_sent_words[t], curr_sent_words[j],curr_sent_pos[j])] )
    #                         two_words_two_pos_list.append((curr_sent_words[j], curr_sent_words[t],curr_sent_pos[j],curr_sent_pos[t]))
    #             curr_sent_words = ['ROOT']
    #             curr_sent_pos   = ['ROOT']
    #         else:
    #             curr_sent_words.append(word_list[i])
    #             curr_sent_pos.append(pos_list[i])
    #     self.h_pos_feat_list = list(set(h_pos_feat_list))
    #     self.m_pos_feat_list = list(set(m_pos_feat_list))
    #     self.h_word_feat_list = list(set(h_word_feat_list))
    #     self.m_word_feat_list = list(set(m_word_feat_list))
    #     self.word_pos_pair_list = list(set(word_pos_pair_list))
    #     self.pos_pair_list = list(set(pos_pair_list))
    #     self.word_pairs_list = list(set(word_pairs_list))
    #     self.word_two_pos_list = list(set(word_two_pos_list))
    #     self.pos_two_words_list = list(set(pos_two_words_list))
    #     self.two_words_two_pos_list = list(set(two_words_two_pos_list))
    #
    #     return

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
            print_err = False):
        # for current word and potential head returns the feature indexes
        feature_idx_list = []
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
            try:
                if feature_kind not in [2,3,5,6]:
                    feat_vars = (tuple(feat_vars), (curr_idx, head_idx))
                else:
                    feat_vars = (feat_vars[0], (curr_idx, head_idx))
                feature_idx_list.append(self.actual_feature_dict[feature_kind]['idx_mapping'][feat_vars])
            except Exception as e:
                if print_err == True:
                    pass
                    # print ('feature_kind:' + str(feature_kind) + ' ' + str(feat_vars) + ' Not in train')

        return feature_idx_list