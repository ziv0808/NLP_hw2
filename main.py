import time
import pickle
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
from utils import create_comp_flie


NUM_OF_PERCEPTRON_STEPS = 80
# if BASIC_MODEL = False then the complex model will be created
BASIC_MODEL = False
# if TRAIN_WITH_MST = False then for training the perceptron will use the greedy method - faster
TRAIN_WITH_MST = True

max_accuracy = 0.83
# load train file
train_words, train_pos, train_heads = read_file_and_preprocess('train.labeled', include_y=True)

# devide word lists into sent lists
sent_word_list, sent_pos_list, sent_head_list = create_sentences_from_word_lists(train_words, train_pos, train_heads)
if True == BASIC_MODEL:
    # create features instance for basic model
    featurs_basic_obj = Featues(train_words, train_pos, train_heads, features_to_include_list=[1,2,3,4,5,6,8,10,13])
else:
    # creates features for the complex model
    featurs_basic_obj = Featues(train_words, train_pos, train_heads, features_to_include_list='ALL')

# init weight vector
basic_feature_weights_vec = np.zeros(featurs_basic_obj.feature_wieghts_len, dtype=np.float64)

# create sentence full graph and for each edge assigns the relevant features list
# also calcs the feature vector for each empiric observation - for optimization
sent_graph_list = []
sent_real_feat_idx = []
sent_graph_edges_feats = []
for m in range(len(sent_word_list)):
    # full graph
    sent_graph_list.append(build_sentence_full_graph(len(sent_word_list[m])))
    # features for correct X + Y
    real_feat_idxs = get_all_feature_idxes_for_sent_and_head(
        featurs_obj=featurs_basic_obj,
        sent_words=sent_word_list[m],
        sent_pos=sent_pos_list[m],
        sent_heads=sent_head_list[m])
    sent_real_feat_idx.append(real_feat_idxs)
    # features assigned to edges
    temp_g_edges = build_graph_features_for_edge(
            G=sent_graph_list[m],
            featurs_obj=featurs_basic_obj,
            sent_words=sent_word_list[m],
            sent_pos=sent_pos_list[m])
    sent_graph_edges_feats.append(temp_g_edges)


# for cui lui
def get_score(h, m):
    return curr_G_wieghts.get((h, m), 0)

t_init = time.time()
######### run perceptron #########
for n in range(NUM_OF_PERCEPTRON_STEPS):
    print ('Starting Perceptron Step - ' + str(n + 1))
    for i in range(len(sent_word_list)):
        curr_G = sent_graph_list[i]
        # creates wieghts to each eadge
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
            # perceptron correction step
            pred_feat_idxs = get_all_feature_idxes_for_sent_and_head(
                featurs_obj=featurs_basic_obj,
                sent_words=sent_word_list[i],
                sent_pos=sent_pos_list[i],
                sent_heads=pred_heads)
            # indexs of real X + Y features
            real_feat_idxs = sent_real_feat_idx[i]

            for feat_idx in real_feat_idxs:
                basic_feature_weights_vec[feat_idx] += 1

            for feat_idx in pred_feat_idxs:
                basic_feature_weights_vec[feat_idx] -= 1

    print ('Finised step after ' + str(time.time() - t_init) + ' Seconds from begining ...')
    if (n+1)%20 == 0:
        # calculating train error
        pred_train_heads_list = []
        for k in range(len(sent_word_list)):
            curr_G = sent_graph_list[k]
            curr_G_wieghts = turn_edge_feats_to_wights(
                        edge_dict=sent_graph_edges_feats[k],
                        wights_vec=basic_feature_weights_vec)
            graph = Digraph(curr_G, get_score)
            mst = graph.mst()
            pred_heads = convert_chi_lui_output_to_list_of_heads(
                sent_len=len(sent_word_list[k]),
                mst=mst)
            pred_train_heads_list.extend(pred_heads)
        print ('Finised calc train error after ' + str(time.time() - t_init) + ' Seconds from begining ...')

        print ('Train:')
        get_results_accuracy(train_heads[:-1], pred_train_heads_list)

        # test model
        test_words, test_pos, test_heads = read_file_and_preprocess('test.labeled', include_y=True)
        # devide word lists into sent lists
        sent_word_list_test, sent_pos_list_test, sent_head_list_test = create_sentences_from_word_lists(test_words, test_pos, test_heads)

        t_test = time.time()
        pred_test_heads_list = []
        for j in range(len(sent_word_list_test)):
            curr_G = build_sentence_full_graph(len(sent_word_list_test[j]))
            curr_G_wieghts = build_graph_wieghts_for_sent(
                G=curr_G,
                wights_vec=basic_feature_weights_vec,
                featurs_obj=featurs_basic_obj,
                sent_words=sent_word_list_test[j],
                sent_pos=sent_pos_list_test[j])
            graph = Digraph(curr_G, get_score)
            mst = graph.mst()
            pred_heads = convert_chi_lui_output_to_list_of_heads(
                sent_len=len(sent_word_list_test[j]),
                mst=mst)
            pred_test_heads_list.extend(pred_heads)
        print ('Finised test inference in ' + str(time.time() - t_test) + ' Seconds')
        print ('Test:')
        score = get_results_accuracy(test_heads[:-1], pred_test_heads_list)
        if score > max_accuracy:
            print ('Tagging comp file...')
            # compatition files tagging
            # comp model
            comp_words, comp_pos, empty_heads = read_file_and_preprocess('comp.unlabeled', include_y=False)
            # devide word lists into sent lists
            sent_word_list_comp, sent_pos_list_comp, empty_heads = create_sentences_from_word_lists(comp_words,
                                                                                                    comp_pos)
            t_comp = time.time()
            pred_comp_heads_list = []
            for j in range(len(sent_word_list_comp)):
                curr_G = build_sentence_full_graph(len(sent_word_list_comp[j]))
                curr_G_wieghts = build_graph_wieghts_for_sent(
                    G=curr_G,
                    wights_vec=basic_feature_weights_vec,
                    featurs_obj=featurs_basic_obj,
                    sent_words=sent_word_list_comp[j],
                    sent_pos=sent_pos_list_comp[j])
                graph = Digraph(curr_G, get_score)
                mst = graph.mst()
                pred_heads = convert_chi_lui_output_to_list_of_heads(
                    sent_len=len(sent_word_list_comp[j]),
                    mst=mst)
                pred_comp_heads_list.extend(pred_heads)
            print('Finised comp inference in ' + str(time.time() - t_comp) + ' Seconds')

            create_comp_flie(comp_words=comp_words,
                             comp_pos=comp_pos,
                             comp_heads=pred_comp_heads_list,
                             basic=BASIC_MODEL)
            if BASIC_MODEL == True:
                model_name = 'Basic'
            else:
                model_name = 'Complicated'
            # saving model features + feature weights
            with open(model_name + '_model_features.pkl', 'wb') as output:
                pickle.dump(featurs_basic_obj, output, pickle.HIGHEST_PROTOCOL)
            with open(model_name + '_model_wieghts.pkl', 'wb') as output:
                pickle.dump(basic_feature_weights_vec, output, pickle.HIGHEST_PROTOCOL)


