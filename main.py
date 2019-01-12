import time
import numpy as np
from Features import Featues
from chu_liu import Digraph
from preprocess import read_file_and_preprocess
from utils import create_sentences_from_word_lists
from utils import build_sentence_full_graph
from utils import build_graph_wieghts_for_sent
from utils import convert_chi_lui_output_to_list_of_heads
from utils import get_all_feature_idxes_for_sent_and_head
from utils import build_graph_features_for_edge
from utils import turn_edge_feats_to_wights
from utils import get_results_accuracy


NUM_OF_PERCEPTRON_STEPS = 30
# if BASIC_MODEL = False then the complex model will be created
BASIC_MODEL = False
# if TRAIN_WITH_MST = False then for training the perceptron will use the greedy method - faster
TRAIN_WITH_MST = True

# load train file
train_words, train_pos, train_heads = read_file_and_preprocess('train.labeled', include_y=True)

# devide word lists into sent lists
sent_word_list, sent_pos_list, sent_head_list = create_sentences_from_word_lists(train_words, train_pos, train_heads)
if True == BASIC_MODEL:
    # create features instance for basic model
    featurs_basic_obj = Featues(train_words, train_pos, train_heads, features_to_include_list=[1,2,3,4,5,6,8,10,13])
else:
    featurs_basic_obj = Featues(train_words, train_pos, train_heads, features_to_include_list='ALL')

# init weight vector
basic_feature_weights_vec = np.zeros(featurs_basic_obj.feature_wieghts_len, dtype=np.float64)

# create sentence full G list and feature list
sent_graph_list = []
sent_real_feat_idx = []
sent_graph_edges_feats = []
for m in range(len(sent_word_list)):
    sent_graph_list.append(build_sentence_full_graph(len(sent_word_list[m])))
    real_feat_idxs = get_all_feature_idxes_for_sent_and_head(
        featurs_obj=featurs_basic_obj,
        sent_words=sent_word_list[m],
        sent_pos=sent_pos_list[m],
        sent_heads=sent_head_list[m])
    sent_real_feat_idx.append(real_feat_idxs)
    temp_g_edges = build_graph_features_for_edge(
            G=sent_graph_list[m],
            featurs_obj=featurs_basic_obj,
            sent_words=sent_word_list[m],
            sent_pos=sent_pos_list[m])
    sent_graph_edges_feats.append(temp_g_edges)



def get_score(h, m):
    return curr_G_wieghts.get((h, m), 0)

t_init = time.time()
######### run perceptron #########
for n in range(NUM_OF_PERCEPTRON_STEPS):
    print ('Starting Perceptron Step - ' + str(n + 1))
    for i in range(len(sent_word_list)):
        curr_G = sent_graph_list[i]
        curr_G_wieghts = turn_edge_feats_to_wights(
                edge_dict=sent_graph_edges_feats[i],
                wights_vec=basic_feature_weights_vec)
        graph = Digraph(curr_G, get_score)
        if True == TRAIN_WITH_MST:
            mst = graph.mst()
        else:
            mst = graph.greedy()
        pred_heads = convert_chi_lui_output_to_list_of_heads(
                sent_len=len(sent_word_list[i]),
                mst=mst)
        if len(pred_heads) != len(sent_head_list[i]):
            print ('Perceptron prediction problem at sent ' + str(i) + ' pred len ' + str(pred_heads) + ' real heads len' + str(sent_head_list[i]))
        if tuple(pred_heads) != tuple(sent_head_list[i]):
            pred_feat_idxs = get_all_feature_idxes_for_sent_and_head(
                featurs_obj=featurs_basic_obj,
                sent_words=sent_word_list[i],
                sent_pos=sent_pos_list[i],
                sent_heads=pred_heads)

            real_feat_idxs = sent_real_feat_idx[i]

            for feat_idx in real_feat_idxs:
                basic_feature_weights_vec[feat_idx] += 1

            for feat_idx in pred_feat_idxs:
                basic_feature_weights_vec[feat_idx] -= 1
    print ('Finised step after ' + str(time.time() - t_init) + ' Seconds from begining ...')


# calculating train error
pred_train_heads_list = []
for i in range(len(sent_word_list)):
    curr_G = sent_graph_list[i]
    curr_G_wieghts = turn_edge_feats_to_wights(
                edge_dict=sent_graph_edges_feats[i],
                wights_vec=basic_feature_weights_vec)
    graph = Digraph(curr_G, get_score)
    mst = graph.mst()
    pred_heads = convert_chi_lui_output_to_list_of_heads(
        sent_len=len(sent_word_list[i]),
        mst=mst)
    pred_train_heads_list.extend(pred_heads)
print ('Finised calc train error after ' + str(time.time() - t_init) + ' Seconds from begining ...')

print ('Train:')
get_results_accuracy(train_heads[:-1], pred_train_heads_list)

# test model
test_words, test_pos, test_heads = read_file_and_preprocess('test.labeled', include_y=True)
# devide word lists into sent lists
sent_word_list_test, sent_pos_list_test, sent_head_list_test = create_sentences_from_word_lists(test_words, test_pos, test_heads)

pred_test_heads_list = []
for i in range(len(sent_word_list_test)):
    curr_G = build_sentence_full_graph(len(sent_word_list_test[i]))
    curr_G_wieghts = build_graph_wieghts_for_sent(
        G=curr_G,
        wights_vec=basic_feature_weights_vec,
        featurs_obj=featurs_basic_obj,
        sent_words=sent_word_list_test[i],
        sent_pos=sent_pos_list_test[i])
    graph = Digraph(curr_G, get_score)
    mst = graph.mst()
    pred_heads = convert_chi_lui_output_to_list_of_heads(
        sent_len=len(sent_word_list_test[i]),
        mst=mst)
    pred_test_heads_list.extend(pred_heads)

print ('Test:')
get_results_accuracy(test_heads[:-1], pred_test_heads_list)





