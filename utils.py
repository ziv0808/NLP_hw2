import numpy as np

def create_sentences_from_word_lists(
        word_list,
        pos_list,
        head_list = None):
    # gets list of words and list of pos and returns list of sentences (word lists)
    # and list of pos lists for the sentences
    sentence_words_list = []
    sentence_pos_list   = []
    sentence_head_list  = []

    curr_sent_words = []
    curr_sent_pos   = []
    curr_sent_heads  = []

    for i in range(len(word_list)):
        if ((word_list[i] == 'ROOT') and (i != 0)) or (i == len(word_list) - 1):
            if (i == len(word_list) - 1) and (word_list[i] != 'ROOT'):
                curr_sent_words.append(word_list[i])
                curr_sent_pos.append(pos_list[i])
                if head_list is not None:
                    curr_sent_heads.append(head_list[i])

            sentence_words_list.append(curr_sent_words)
            sentence_pos_list.append(curr_sent_pos)
            if head_list is not None:
                sentence_head_list.append(curr_sent_heads)
                curr_sent_heads = ['ROOT']

            curr_sent_words = ['ROOT']
            curr_sent_pos = ['ROOT']

        else:
            curr_sent_words.append(word_list[i])
            curr_sent_pos.append(pos_list[i])
            if head_list is not None:
                curr_sent_heads.append(head_list[i])

    return sentence_words_list, sentence_pos_list, sentence_head_list

def build_sentence_full_graph(
        sent_len):
    # returns for sentence all graphs potential nodes for use of cui liu
    G = {}
    G[0] = list(range(1,sent_len))
    for i in range(1,sent_len):
        G[i] = list(range(1,sent_len))
        G[i].remove(i)
    return G

def build_graph_wieghts_for_sent(
        G,
        wights_vec,
        featurs_obj,
        sent_words,
        sent_pos):
    # returns for the current sentence graph all edges and thier wieght for cui liu
    G_wieghts = {}
    for head_idx, word_idx_list in G.items():
        for curr_word_idx in word_idx_list:
            curr_edge_feature_idxes_list = featurs_obj.get_feature_indexs(
                    head_word = sent_words[head_idx],
                    head_pos  = sent_pos[head_idx],
                    curr_word = sent_words[curr_word_idx],
                    curr_pos  = sent_pos[curr_word_idx],
                    head_idx  = head_idx,
                    curr_idx  = curr_word_idx)
            if len(curr_edge_feature_idxes_list) == 0:
                G_wieghts[(head_idx, curr_word_idx)] = 0
            else:
                G_wieghts[(head_idx, curr_word_idx)] = np.sum(wights_vec[curr_edge_feature_idxes_list])
    return G_wieghts

def build_graph_features_for_edge(
        G,
        featurs_obj,
        sent_words,
        sent_pos):
    # returns for the current sentence graph all edges and their list of features
    G_edge_features = {}
    for head_idx, word_idx_list in G.items():
        for curr_word_idx in word_idx_list:
            curr_edge_feature_idxes_list = featurs_obj.get_feature_indexs(
                head_word=sent_words[head_idx],
                head_pos=sent_pos[head_idx],
                curr_word=sent_words[curr_word_idx],
                curr_pos=sent_pos[curr_word_idx],
                head_idx=head_idx,
                curr_idx=curr_word_idx,
                print_err=True)
            G_edge_features[(head_idx, curr_word_idx)] = curr_edge_feature_idxes_list
    return G_edge_features

def turn_edge_feats_to_wights(
        edge_dict,
        wights_vec):

    G_wieghts = {}
    for key in edge_dict:
        if len(edge_dict[key]) == 0:
            G_wieghts[key] = 0
        else:
            G_wieghts[key] = np.sum(wights_vec[edge_dict[key]])
    return G_wieghts


def convert_chi_lui_output_to_list_of_heads(
        sent_len,
        mst):
    # converts mst into head list
    head_list = ['ROOT']*sent_len
    for head_idx, word_idx_list in mst.successors.items():
        for curr_word_idx in word_idx_list:
            head_list[curr_word_idx] = head_idx
    if 'ROOT' in head_list[1:]:
        print ("Prediction Problem")

    return head_list

def get_all_feature_idxes_for_sent_and_head(
        featurs_obj,
        sent_words,
        sent_pos,
        sent_heads):
    # for sentence and itsheads get all the feature indexs (can be multiple of same feature idx)
    idx_list = []
    for i in range(1, len(sent_words)):
        curr_edge_feature_idxes_list = featurs_obj.get_feature_indexs(
            head_word=sent_words[int(sent_heads[i])],
            head_pos=sent_pos[sent_heads[i]],
            curr_word=sent_words[i],
            curr_pos=sent_pos[i],
            head_idx=int(sent_heads[i]),
            curr_idx=i)
        idx_list.extend(curr_edge_feature_idxes_list )
    return idx_list


def get_results_accuracy(
        actual_heads,
        pred_heads):
    if len(pred_heads) != len(actual_heads):
        print('Prediction len problem actual heads len:' + str(len(actual_heads)) + ' pred heads len:' + str(
            len(pred_heads)))
    total = 0.0
    correct = 0.0
    for i in range(len(pred_heads)):
        if actual_heads[i] != 'ROOT':
            total += 1.0
            if int(actual_heads[i]) == int(pred_heads[i]):
                correct += 1.0

    print ('Acuurcay :' + str(correct/float(total)))


