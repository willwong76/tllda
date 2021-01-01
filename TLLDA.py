# -*- coding: utf-8 -*-

"""

Twin labeled LDA
Wang Wei 2019-10-5
Based on Fishe LDA Collapse Gibbs Sampling

"""

from numpy.core._multiarray_umath import ndarray
import os.path as osp
import multiprocessing
import numpy as np

'''
###############################################################################
                             读取训练和测试数据
###############################################################################
'''

# 雅虎数据  # mat文件的读取路径（需要修改）
from scipy.io import loadmat

import time

my_dataset = r'Yahoo_Health_Split01'  # 数据集
root = r'/Users/wangwei/Documents/DependencyLDA-master/EvaluationDatasets/'

my_data = loadmat(
    osp.join(root, my_dataset, '{}.mat'.format(my_dataset))
)

path_phi = osp.join(root, my_dataset, '{}LHLDA_phi.npy'.format(my_dataset))  # 训练结果 phi 的存储位置
path_nw = osp.join(root, my_dataset, '{}LHLDA_n_w_out.npy'.format(my_dataset))  # 训练结果 n_w_out 的存储位置
path_theta_test = osp.join(root, my_dataset, '{}theta_test.npz'.format(my_dataset))  # 测试结果 theta_test 的存储位置
path_g_phi = osp.join(root, my_dataset, '{}LHLDA_g_phi.npy'.format(my_dataset))  # 训练结果 g_phi 的存储位置
path_g_nw = osp.join(root, my_dataset, '{}LHLDA_g_n_w_out.npy'.format(my_dataset))  # 训练结果 g_n_w_out 的存储位置
path_m_gt = osp.join(root, my_dataset, '{}LHLDA_m_gt.npy'.format(my_dataset))  # 训练结果 m_gt 的存储位置
path_para = osp.join(root, my_dataset, '{}LHLDA_para.npy'.format(my_dataset))  # 训练迭代的alpha存储位置
path_group_test = osp.join(root, my_dataset, '{}group_test.npz'.format(my_dataset))

# 训练数据(不可修改)
train_id_topic4doc = my_data['traindata'][0][0]['cdidx']  # topic的文章索引
train_id_topic4topic = my_data['traindata'][0][0]['cidx']  # topic的topic索引
train_id_words4doc = my_data['traindata'][0][0]['wdidx']  # words的文章索引
train_id_words4words = my_data['traindata'][0][0]['widx']  # words的words索引

# 测试数据(不可修改)
test_id_topic4doc_all = my_data['testdata'][0][0]['cdidx']  # topic的文章索引
test_id_topic4topic = my_data['testdata'][0][0]['cidx']  # topic的topic索引
test_id_words4doc_all = my_data['testdata'][0][0]['wdidx']  # words的文章索引
test_id_words4words_all = my_data['testdata'][0][0]['widx']  # words的words索引

'''
其它修改（为了操作方便我整合了部分参数设置到此处，后面相应部分进行了注释）
'''

max_iter = 5 # 训练迭代次数
alpha_doc = 50  # 初始化的文章 level 的 alpha
alpha_group = 50  # 初始化的组的alpha
alpha_doc_test = 2  # 测试时的topic alpha，如果用传递模式，会被更新
alpha_group_test = 1  # 测试时的组alpha，如果用传递模式，会被更新
use_train_para_doc = 0  # 使用训练阶段迭代的alpha_doc参数
use_train_para_group = 0  # 使用训练阶段迭代的alpha_doc参数
auto_gap = 1  # 参数更新间隔
auto_pand = 0  # 是否更新alpha和alpha_1的控制参数，若都不更新=0，若只更新alpha=1, 若只更新alpha_1=2, 若都要更新=3
auto_gand = 0  # 是否更新g_alpha和g_alpha_1的控制参数，若都不更新=0，若只更新alpha=1, 若只更新alpha_1=2, 若都要更新=3
test_iter = 10  # 测试的迭代次数
LT = 2  # LT 隐主题数量，或者是功能词的主题
LT_prop = 0.99  # 隐主题占的比例（在先验之外的）
g_LT = 300  # 隐藏组模式，由于出现的组是少数，2^K个组，所以数字应该比较大
LProp = 0.99 # 先验标签对应主题所占比例
g_LProp = 0.1  # 先验分组所占比例
Training_Chain_Num = 3
LongLimit = 3  # 过长文章判断阈值
UsePhi = 1  # 在测试时是直接用Phi套用更新，还是继续计数
phi_refresh = 1000  # 测试时phi的更新频率
num_want = 0  # 测试样本数量，若不需要修改设置为 0
Batch_Mode = 1  # 是否采用批次预测
Test_Batch = 100  # 每批次预测文本数量
Batch_Refresh = 0  # 每批次结束后是否更新模型
TrainTestOutput = 7  # 0b100:训练，0b010:测试，0b001:输出结果  7:全做  3：测试和输出  1:输出
MultiChainTestMode = 1  # 多个链测试模式
Sample_Num = 3  # 采样次数
Sample_Gap = 5  # 测试时的采样间隔
Test_Chain_Num = 10  # 测试链数量
UseGMode = 1  # 使用G模式
FMode = 1  # 训练时考虑标签出现的频率，再套一层Dir模型
alpha_corpus = 50
Gconv = 1  # 使用G模式时，是否融合m_gt，0：不融合，取最大，1：全融合，2：大于gap的融合
MultiGTest = 6  # 测试G也采用多条链

entropy_break = 'no'  # 测试时是否引入熵跳出机制，'yes' or 'no'
entropy_lim = 0.5  # 测试时熵跳出机制的阈值
NeedRenewGroup = 1 #是否需要重新计算Group和文档关系，1需要，0从存储中获取
SimpleStatistics = 0 #不计算Margin和Is Error
multi_test_mode = 0  #0 Proportioanl   1 calibrated  2 BEP

if __name__ == '__main__':
    localtime = time.asctime(time.localtime(time.time()))
    print('开始时间为 :', localtime)

    L = np.size(my_data['clabels']) + LT  # topic 的数量
    V = np.size(my_data['wlabels'])  # words 的数量(语料库)
    M = int(max(train_id_topic4doc)[0])  # 训练文本数量

    print('数据集：', my_dataset)
    print('训练标签数量', L)
    print('训练单词数量', V)
    print('训练文本数量', M)

    # 计算文章-主题矩阵 和 文章-词矩阵（计数）
    doc2topic_ind = np.zeros((M, L), dtype='int')

    # 文章-主题示性矩阵
    for i in range(len(train_id_topic4doc)):
        doc2topic_ind[train_id_topic4doc[i][0] - 1, train_id_topic4topic[i][0] - 1] = 1

    beta = 0.01  # 词先验分布参数
    g_beta = 0.01  # 训练组时的先验分布
#####################################################################

    Total_label = doc2topic_ind.sum()

    Med_Topic_num = int(round(np.median(doc2topic_ind.sum(axis=1), axis=0)))
    Avg_Topic_num = int(round(Total_label / M))

    print('文档平均标签数', Total_label / M)
    print('标签数中位值', Med_Topic_num)

    K = L  # 设置实际topic的数量，可以大于L
    mu = 1 / K * np.ones((K,))  # class 层使用的向量

    # 计算group的相应部分
    path_group_test = osp.join(root, my_dataset, '{}group_test.npz'.format(my_dataset))
    if NeedRenewGroup == 1:
        doc2NeedFlag = np.ones(M)  # 还未归入组的标志
        doc2NeedFlag[:] = -1  # 文档属于哪一组
        group_topic_ind = np.zeros(((g_LT + M), K), dtype="int")  # group对应的topic组合
        group_flag = np.zeros((g_LT + M))  # 该组是否有训练的文档
        G_num = 0;  # 训练数据的组数
        group_doc_num = np.zeros((g_LT + M))
        for i in range(M):
            if (i % 1000 == 0) and (i > 0):
                print('计算文档和组的关系，已处理文档数：', i)
            if (doc2NeedFlag[i] == -1):
                for j in range(M):
                    if group_flag[j] == 0:
                        G_num = j
                        break
                    if (doc2topic_ind[i] == group_topic_ind[j]).all():
                        doc2NeedFlag[i] = j
                        group_doc_num[j] += 1
                if (doc2NeedFlag[i] == -1):  # 没找到
                    group_topic_ind[G_num] = doc2topic_ind[i]
                    group_flag[G_num] = 1
                    group_doc_num[G_num] += 1
                    doc2NeedFlag[i] = G_num

        doc2NeedFlag.astype(int)

        np.savez(path_group_test, doc2NeedFlag=doc2NeedFlag, group_topic_ind=group_topic_ind.T, group_flag=group_flag, G_num=G_num, group_doc_num=group_doc_num)
        print('存储文档和组的关系')
    else:
        group_file = np.load(path_group_test)
        doc2NeedFlag = group_file['doc2NeedFlag']
        group_topic_ind = group_file['group_topic_ind']
        group_flag = group_file['group_flag']
        G_num = group_file['G_num']
        group_doc_num = group_file['group_doc_num']
        print('读取文档和组的关系')


    print('总分组数', G_num)
    print('每个组的平均文档%f，平均标签%f' % ((group_doc_num.sum() / G_num), group_topic_ind.sum() / G_num))
    print('每个主题的文章分布情况：', doc2topic_ind.sum(axis=0))
    print('每个主题的组分布情况：', group_topic_ind.sum(axis=0))

    group_avg_num = group_doc_num.sum() / G_num
    # 计算每个文档的主题数

    # 计算初始化 class 层的 alpha,初始化mu=1/K,标签对应主题所占份额为LProp，即K-KLT/K
    KLT = K * (1 - LProp)  # 标签对应的主题占的份额为 （K-KLT）/K

    alpha_class = K * group_avg_num * Avg_Topic_num * (1 - LProp) / (K * LProp - Avg_Topic_num)

    # 使用频率
    topic_label_num = doc2topic_ind.sum(axis=0)
    if FMode == 1:
        mu = (topic_label_num + alpha_corpus * mu) / (topic_label_num.sum() + alpha_corpus)

    latent_mu = LT * ( alpha_corpus / K) / (topic_label_num.sum() + alpha_corpus)

    # 根据参数调整隐主题所占比例
    for i in range(LT):
        mu[K - i - 1] = LT_prop / LT
    for i in range(L-LT):
        mu[i]*= ((1-LT_prop)/(1-latent_mu))

    print('alpha_corpus, mu', alpha_corpus, mu)

    print('初始化 alpha, alpha_1, mu:', alpha_doc, alpha_class, (1 / K))

    G = G_num + g_LT  # 准备输出的组数，g_LT>0,因为出现的组毕竟是少数

    g_alpha_class = (1 - g_LProp) / (g_LProp - 1 / G)

    g_mu = 1 / G * np.ones((G,))  # g_class 层使用的向量

    print('初始化 g_alpha, g_alpha_1, mu:', alpha_group, g_alpha_class, (1 / G))

'''
###############################################################################
                         LHLDA 的 m_gt 获取函数定义，用论文中最短路径假设
###############################################################################
'''


def LHLDA_mgt_get2(doc_label_num, alpha_1, mu, group_num, label_vec=[]):
    '''
    参数说明
    '''
    # doc_label_num：文章-标签数量
    # alpha_1：float；
    # mu：mu，1d-array；
    # group_num：迭代文章所在组的文章数
    topic_num_count = label_vec.copy()

    m_gt_yes = (group_num * topic_num_count + alpha_1 * mu) / (doc_label_num * group_num + alpha_1)
    m_gt_no = alpha_1 * mu / (doc_label_num * group_num + alpha_1)

    return m_gt_yes, m_gt_no


'''
###############################################################################
                         LHLDA 的 g_m_gt 获取函数定义
###############################################################################
'''


def LHLDA_g_mgt_get2(G, group, g_alpha_1, g_mu, group_flag):
    '''
    参数说明
    '''
    # G: 总的分组个数
    # group：分组号
    # prop：先验分组所占比例

    g_mgt = np.zeros(G)
    if group_flag[group] == 1:
        g_ind = np.zeros(G)
        g_ind[group] = 1
        g_mgt = (g_ind + g_alpha_1 * g_mu) / (1 + g_alpha_1)

    else:
        g_mgt[:] = 1 / G

    return g_mgt


'''
###############################################################################
                           LHLDA 主程序 (train)
###############################################################################
'''





def LHLDA_Yahoo_train(doc2topic_ind, train_id_words4doc, train_id_words4words, M, V, K, G, alpha_doc, alpha_class,
                      alpha_group, g_alpha_class, mu, g_mu, beta, g_beta, max_iter, auto_gap, auto_pand, doc2NeedFlag,
                      group_flag, group_doc_num, group_topic_ind):
    '''
    参数说明
    '''
    # doc2topic_ind：文章-主题示性矩阵，（初始化）m_gt的计算参数
    # train_id_words4doc：文章索引
    # train_id_words4words：词索引
    # M：文章数目
    # V：语料库数目
    # K：topic数目，m_gt的计算参数
    # alpha_doc：文章级别的alpha
    # alpha_class：topic级别的alpha_1，m_gt的计算参数
    # mu：topic级别的先验分布，m_gt的计算参数
    # beta：主题——词先验，int
    # g_beta: 分组训练时的beta
    # max_iter：最大迭代次数
    # auto_gap：更新 alpha 和 alpha_1 的间隔
    # auto_pand：是否更新 alpha 和 alpha_1
    # G: 组数
    # alpha_group: 用于训练组的alpha
    # g_prop： 训练group时，先验组所占比例
    # doc2NeedFlag： 文章属于哪个组

    ###########################################################################
    # 计算每个训练样本所包含的词的数量
    ###########################################################################

    import numpy as np
    import os
    np.random.seed()
    n_d_sum = np.zeros((M, 1), dtype="int")  # 储存文档中词的总数
    for i in range(M):
        n_d_sum[i] = np.sum(train_id_words4doc == [i + 1])

    ###########################################################################
    # 构建数据存储数组
    ###########################################################################


    Z_1 = np.array([[0 for j in range(int(n_d_sum[i]))] for i in range(M)])  # 生成用于储存一级主题分配矩阵

    g_Z_1 = np.array([[0 for j in range(int(n_d_sum[i]))] for i in range(M)])  # 生成用于储存分组号分配矩阵

    n_d_1 = np.zeros((M, K), dtype="int")  # 储存文档中每个一级主题词的个数
    n_w_1 = np.zeros((V, K), dtype="int")  # 储存每个词在一级主题中的分布（频数）
    n_w_1_sum = np.zeros((1, K), dtype="int")  # 储存每个一级主题中词的个数

    g_n_d_1 = np.zeros((M, G), dtype="int")  # 储存文档中每个分组的词的个数
    g_n_w_1 = np.zeros((V, G), dtype="int")  # 储存每个词在分组中的分布（频数）
    g_n_w_1_sum = np.zeros((1, G), dtype="int")  # 储存每个分组中词的个数

    ###############################################################################
    # Group 信息计算（train use）
    ###############################################################################

    # 计算所有样本的 Group 索引（真实）
    if auto_pand != 0:
        print('计算所有样本的 Group 索引')
        Group_index = np.zeros((M,), dtype='int')
        for i in range(M):
            for j in range(K):
                Group_index[i] += doc2topic_ind[i, j] * (2 ** j)

        # 计算所有样本的 Group 索引（相对）
        Group_num_set = list(np.sort(np.array(list(set(list(Group_index))))))

        Group_index_xd = np.zeros((M,), dtype='int')
        for i in range(M):
            Group_index_xd[i] = Group_num_set.index(Group_index[i])

    ###########################################################################
    # 初始化主题选择+数据统计
    ###########################################################################

    for i in range(M):

        if i % 200 == 0:
            print('正在初始化第 %d 个训练样本' % (i))

        doc_topic_now_1 = doc2topic_ind[i, :]  # 当前文章的主题（可能是多个1）

        m_gt_yes, m_gt_no = LHLDA_mgt_get2(doc_topic_now_1.sum(), alpha_class, mu, group_doc_num[int(doc2NeedFlag[i])],
                                           doc_topic_now_1)

        m_c_now_1 = m_gt_yes

        m_c_now_1 = m_c_now_1 / m_c_now_1.sum()  # 归一化

        g_m_gt = LHLDA_g_mgt_get2(G, int(doc2NeedFlag[i]), g_alpha_class, g_mu, group_flag)

        words_begin_now = int(np.sum(n_d_sum[0:i]))  # 当前文章词索引的起始位置

        for j in range(int(n_d_sum[i])):

            myword_ind = int(train_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值
            topic_4myword = np.argmax(np.random.multinomial(1, m_c_now_1))  # 采用先验概率来产生初始化的主题
            Z_1[i][j] = topic_4myword  # 记录当前词的主题

            # 更新对应频次统计量
            n_d_1[i, topic_4myword] += 1  # 更新 n_d_1
            n_w_1[myword_ind, topic_4myword] += 1  # 更新 n_w_1
            n_w_1_sum[0, topic_4myword] += 1  # 更新 n_w_1_sum

            if UseGMode == 1:
                group_4myword = np.argmax(np.random.multinomial(1, g_m_gt))  # 采用先验概率来产生初始化的分组
                g_Z_1[i][j] = group_4myword  # 记录当前词的g_分组

                # 更新对应频次统计量
                g_n_d_1[i, group_4myword] += 1  # 更新 g_n_d_1
                g_n_w_1[myword_ind, group_4myword] += 1  # 更新 g_n_w_1
                g_n_w_1_sum[0, group_4myword] += 1  # 更新 g_n_w_1_sum




    ###########################################################################
    # 迭代主过程
    ###########################################################################

    t = 1  # 迭代次数初始化
    alpha_all = np.zeros((max_iter,))  # 存储每次迭代的alpha
    alpha_1_all = np.zeros((max_iter,))  # 储存每次迭代的alpha_1
    en_all = np.zeros((max_iter,))  # 储存每次迭代的熵

    while t <= max_iter:

        print('开始第 %d 次迭代...' % (t))
        alpha_all[t - 1] = alpha_doc  # 记录当前 alpha
        alpha_1_all[t - 1] = alpha_class  # 记录当前 alpha_1

        #######################################################################
        # 对每一篇文章的词重新选择主题
        #######################################################################

        for i in range(M):
            if i % 200 == 0:
                print('正在训练第 %d 个训练样本' % (i))

            doc_topic_now_1 = doc2topic_ind[i, :]  # 当前文章的主题（可能是多个1）

            m_gt_yes, m_gt_no = LHLDA_mgt_get2(doc_topic_now_1.sum(), alpha_class, mu,
                                               group_doc_num[int(doc2NeedFlag[i])], doc_topic_now_1)

            m_c_now_1 = m_gt_yes  # 计算当前文章的权值
            m_c_now_1 = m_c_now_1 / m_c_now_1.sum()  # 归一化

            g_m_gt = LHLDA_g_mgt_get2(G, int(doc2NeedFlag[i]), g_alpha_class, g_mu, group_flag)  # 计算当前文章的g_m_t

            words_begin_now = int(np.sum(n_d_sum[0:i]))  # 当前文章词索引的起始位置

            for j in range(int(n_d_sum[i])):
                myword_ind = int(train_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值

                # --------------------------------------------------------------
                topic_1 = Z_1[i][j]  # 提取一级主题
                n_d_1[i, topic_1] -= 1  # 更新 n_d_1
                n_w_1[myword_ind, topic_1] -= 1  # 更新 n_w_1
                n_w_1_sum[0, topic_1] -= 1  # 更新 n_w_sum_1
                n_d_sum[i] -= 1  # 更新 n_d_sum
                # --------------------------------------------------------------
                P_1 = ((n_w_1[myword_ind, :] + beta) / (n_w_1_sum + beta * V)) * (
                        n_d_1[i, :] + (alpha_doc * m_c_now_1)) / (n_d_sum[i, 0] + alpha_doc)  # 词的主题分布概率
                P_1_std = np.squeeze(np.asarray(P_1 / P_1.sum()))  # 归一化
                # --------------------------------------------------------------
                topic_1 = np.argmax(np.random.multinomial(1, P_1_std))  # 产生一个新的主题
                Z_1[i][j] = topic_1  # 把新的 topic_1 赋值给 Z_1
                n_d_1[i, topic_1] += 1  # 更新 n_d_1
                n_w_1[myword_ind, topic_1] += 1  # 更新 n_w_1
                n_w_1_sum[0, topic_1] += 1  # 更新 n_w_sum_1
                n_d_sum[i] += 1  # 更新 n_d_sum

                # 计算组别
                if UseGMode == 1:
                    group_1 = g_Z_1[i][j]  # 提取组
                    g_n_d_1[i, group_1] -= 1  # 更新 g_n_d_1
                    g_n_w_1[myword_ind, group_1] -= 1  # 更新 g_n_w_1
                    g_n_w_1_sum[0, group_1] -= 1  # 更新 g_n_w_sum_1
                    n_d_sum[i] -= 1  # 更新 n_d_sum
                    # --------------------------------------------------------------
                    g_P_1 = ((g_n_w_1[myword_ind, :] + g_beta) / (g_n_w_1_sum + g_beta * V)) * (
                            g_n_d_1[i, :] + (alpha_group * g_m_gt)) / (n_d_sum[i, 0] + alpha_group)  # 词的主题分布概率
                    g_P_1_std = np.squeeze(np.asarray(g_P_1 / g_P_1.sum()))  # 归一化
                    # --------------------------------------------------------------
                    group_1 = np.argmax(np.random.multinomial(1, g_P_1_std))  # 产生一个新的分组
                    g_Z_1[i][j] = group_1  # 把新的 group_1 赋值给 g)_Z_1
                    g_n_d_1[i, group_1] += 1  # 更新 g_n_d_1
                    g_n_w_1[myword_ind, group_1] += 1  # 更新 g_n_w_1
                    g_n_w_1_sum[0, group_1] += 1  # 更新 g_n_w_sum_1
                    n_d_sum[i] += 1  # 更新 n_d_sum

        #######################################################################
        # 定期更新 alpha 和 alpha_1
        #######################################################################
        if t % auto_gap == 0:

            if auto_pand == 0:
                print('跳过参数更新')
            elif auto_pand == 1:
                print('只更新 alpha')
                ###################################################################
                # 更新 alpha
                ###################################################################
                G_r = len(Group_num_set)
                m_gt_all_refesh = np.zeros((G_r, K))
                for i in range(G_r):
                    m_gt_all_refesh[i] = (group_doc_num[i] * group_topic_ind[i] + alpha_class * mu) / (
                            group_topic_ind[i].sum() * group_doc_num[i] + alpha_class)

                S_k1 = 0  # 分子
                S_1 = 0  # 分母
                for j in range(M):

                    m_gt_yes_all = m_gt_all_refesh[Group_index_xd[j]]
                    m_g_t_1 = m_gt_yes_all / m_gt_yes_all.sum()

                    if ((n_d_1[j, :].sum() + alpha_doc - 0.5) / (alpha_doc + 0.5)) > 0:
                        S_1 += (1 / alpha_doc) + np.log((n_d_1[j, :].sum() + alpha_doc - 0.5) / (alpha_doc + 0.5))  # 分母
                        for i in range(K):
                            if n_d_1[j, i] != 0 and m_g_t_1[i] != 0 and (
                                    (n_d_1[j, i] + alpha_doc * m_g_t_1[i] - 0.5) / (alpha_doc * m_g_t_1[i] + 0.5)) > 0:
                                S_k1 += m_g_t_1[i] * ((1 / (alpha_doc * m_g_t_1[i])) + np.log(
                                    (n_d_1[j, i] + alpha_doc * m_g_t_1[i] - 0.5) / (
                                            alpha_doc * m_g_t_1[i] + 0.5)))  # 分子

                alpha_doc = alpha_doc * (S_k1 / S_1)  # 更新


            elif auto_pand == 2:
                print('只更新 alpha_1 和 mu')
                ###################################################################
                # 更新 alpha_1 和 mu
                ###################################################################
                n_g_1 = np.zeros((len(Group_num_set), K), dtype='int')
                for i in range(M):
                    n_g_1[Group_index_xd[i], :] += n_d_1[i, :]

                S_k1 = np.zeros((K,))  # 储存计算后的 （分子）
                S_1 = 0  # 分母

                for j in range(len(Group_num_set)):
                    if ((n_g_1[j, :].sum() + alpha_class - 0.5) / (alpha_class + 0.5)) > 0:
                        S_1 += (1 / alpha_class) + np.log(
                            (n_g_1[j, :].sum() + alpha_class - 0.5) / (alpha_class + 0.5))  # 分母
                        for i in range(K):
                            if n_g_1[j, i] != 0 and (
                                    (n_g_1[j, i] + alpha_class * mu[i] - 0.5) / (alpha_class * mu[i] + 0.5)) > 0:
                                S_k1[i] += (1 / (alpha_class * mu[i])) + np.log(
                                    (n_g_1[j, i] + alpha_class * mu[i] - 0.5) / (alpha_class * mu[i] + 0.5))  # 分子

                alpha_c1_u = alpha_class * mu * (S_k1 / S_1)
                alpha_class = alpha_c1_u.sum()  # 新的
                mu = alpha_c1_u / alpha_class


            elif auto_pand == 3:
                print('同时更新 alpha、alpha_1 和 mu')
                ###################################################################
                # 更新 alpha_1 和 mu
                ###################################################################
                n_g_1 = np.zeros((len(Group_num_set), K), dtype='int')
                for i in range(M):
                    n_g_1[Group_index_xd[i], :] += n_d_1[i, :]

                S_k1 = np.zeros((K,))  # 储存计算后的 （分子）
                S_1 = 0  # 分母

                for j in range(len(Group_num_set)):
                    if ((n_g_1[j, :].sum() + alpha_class - 0.5) / (alpha_class + 0.5)) > 0:
                        S_1 += (1 / alpha_class) + np.log(
                            (n_g_1[j, :].sum() + alpha_class - 0.5) / (alpha_class + 0.5))  # 分母
                        for i in range(K):
                            if n_g_1[j, i] != 0 and (
                                    (n_g_1[j, i] + alpha_class * mu[i] - 0.5) / (alpha_class * mu[i] + 0.5)) > 0:
                                S_k1[i] += (1 / (alpha_class * mu[i])) + np.log(
                                    (n_g_1[j, i] + alpha_class * mu[i] - 0.5) / (alpha_class * mu[i] + 0.5))  # 分子

                alpha_c1_u = alpha_class * mu * (S_k1 / S_1)
                alpha_class = alpha_c1_u.sum()  # 新的
                mu = alpha_c1_u / alpha_class
                ###################################################################
                # 更新 alpha
                ###################################################################
                G_r = len(Group_num_set)
                m_gt_all_refesh = np.zeros((G_r, K))
                for i in range(G_r):
                    m_gt_all_refesh[i] = (group_doc_num[i] * group_topic_ind[i] + alpha_class * mu) / (
                            group_topic_ind[i].sum() * group_doc_num[i] + alpha_class)

                S_k1 = 0  # 分子
                S_1 = 0  # 分母
                for j in range(M):

                    m_gt_yes_all = m_gt_all_refesh[Group_index_xd[j]]
                    m_g_t_1 = m_gt_yes_all / m_gt_yes_all.sum()

                    if ((n_d_1[j, :].sum() + alpha_doc - 0.5) / (alpha_doc + 0.5)) > 0:
                        S_1 += (1 / alpha_doc) + np.log((n_d_1[j, :].sum() + alpha_doc - 0.5) / (alpha_doc + 0.5))  # 分母
                        for i in range(K):
                            if n_d_1[j, i] != 0 and m_g_t_1[i] != 0 and (
                                    (n_d_1[j, i] + alpha_doc * m_g_t_1[i] - 0.5) / (alpha_doc * m_g_t_1[i] + 0.5)) > 0:
                                S_k1 += m_g_t_1[i] * ((1 / (alpha_doc * m_g_t_1[i])) + np.log(
                                    (n_d_1[j, i] + alpha_doc * m_g_t_1[i] - 0.5) / (
                                            alpha_doc * m_g_t_1[i] + 0.5)))  # 分子

                alpha_doc = alpha_doc * (S_k1 / S_1)  # 更新

            if auto_gand == 0:
                print('跳过group参数更新')
            if (UseGMode == 1) and ((auto_gand & 2) == 2):  # 更新g_alpha_1和g_mu
                print('G_Mode:更新 g_alpha_1 和 g_mu')
                ###################################################################
                # 更新 g_alpha_1 和 g_mu
                ###################################################################
                g_n_g_1 = np.zeros((G, G), dtype='int')
                for i in range(M):
                    g_n_g_1[int(doc2NeedFlag[i]), :] += g_n_d_1[i, :]

                S_k1 = np.zeros((G,))  # 储存计算后的 （分子）
                S_1 = 0  # 分母

                for j in range(G):
                    if ((g_n_g_1[j, :].sum() + g_alpha_class - 0.5) / (g_alpha_class + 0.5)) > 0:
                        S_1 += (1 / g_alpha_class) + np.log(
                            (g_n_g_1[j, :].sum() + g_alpha_class - 0.5) / (g_alpha_class + 0.5))  # 分母
                        for i in range(K):
                            if g_n_g_1[j, i] != 0 and (
                                    (g_n_g_1[j, i] + g_alpha_class * g_mu[i] - 0.5) / (
                                    g_alpha_class * g_mu[i] + 0.5)) > 0:
                                S_k1[i] += (1 / (g_alpha_class * g_mu[i])) + np.log(
                                    (g_n_g_1[j, i] + g_alpha_class * g_mu[i] - 0.5) / (
                                            g_alpha_class * g_mu[i] + 0.5))  # 分子

                g_alpha_c1_u = g_alpha_class * g_mu * (S_k1 / S_1)
                g_alpha_class = g_alpha_c1_u.sum()  # 新的
                g_mu = g_alpha_c1_u / g_alpha_class
            if (UseGMode == 1) and ((auto_gand & 1) == 1):
                print('更新 g_alpha')
                ###################################################################
                # 更新 g_alpha
                ###################################################################
                g_m_gt_all_refesh = np.zeros((G, K))

                S_k1 = 0  # 分子
                S_1 = 0  # 分母
                for j in range(M):
                    g_m_gt = LHLDA_g_mgt_get2(G, int(doc2NeedFlag[j]), g_alpha_class, g_mu, group_flag)

                    g_m_g_t_1 = g_m_gt / g_m_gt.sum()

                    if ((g_n_d_1[j, :].sum() + alpha_group - 0.5) / (alpha_group + 0.5)) > 0:
                        S_1 += (1 / alpha_group) + np.log(
                            (g_n_d_1[j, :].sum() + alpha_group - 0.5) / (alpha_group + 0.5))  # 分母
                        for i in range(K):
                            if g_n_d_1[j, i] != 0 and m_g_t_1[i] != 0 and (
                                    (g_n_d_1[j, i] + alpha_group * g_m_g_t_1[i] - 0.5) / (
                                    alpha_group * g_m_g_t_1[i] + 0.5)) > 0:
                                S_k1 += g_m_g_t_1[i] * ((1 / (alpha_group * g_m_g_t_1[i])) + np.log(
                                    (g_n_d_1[j, i] + alpha_group * g_m_g_t_1[i] - 0.5) / (
                                            alpha_group * g_m_g_t_1[i] + 0.5)))  # 分子

                alpha_group = alpha_group * (S_k1 / S_1)  # 更新

            print('下一步迭代将使用的 alpha:', alpha_doc)
            print('下一步迭代将使用的 alpha_1:', alpha_class)
            print('下一步迭代将使用的 g_alpha:', alpha_group)
            print('下一步迭代将使用的 g_alpha_1:', g_alpha_class)
            if entropy_break == 'yes':
                print('当前迭代完成后每个文本的平均信息熵为:', entropy_now.sum() / M)

        #######################################################################
        # 计算熵
        #######################################################################

        if entropy_break == 'yes':
            entropy_now = np.zeros((M,))
            for i in range(M):
                for j in range(K):
                    if n_d_1[i, j] != 0:
                        entropy_now[i] += -(n_d_1[i, j] / n_d_1[i, :].sum()) * np.log2(n_d_1[i, j] / n_d_1[i, :].sum())
            en_all[t - 1] = entropy_now.sum() / M  # 记录平均熵


        t += 1



    ###########################################################################
    # 计算 主题-词 矩阵 Phi
    ###########################################################################

    print('生成返回值 phi')

    phi = np.zeros((K, V))
    for i in range(K):
        phi[i, :] = (n_w_1.T[i, :] + beta) / (n_w_1_sum[0, i] + beta * V)



    ###########################################################################
    # 计算 分组相关
    ###########################################################################
    print('生成返回值 g_phi')

    g_phi = np.zeros((G, V))

    m_gt_t = np.zeros((G, K))
    if UseGMode == 1:
        for i in range(G):
            g_phi[i, :] = (g_n_w_1.T[i, :] + g_beta) / (g_n_w_1_sum[0, i] + g_beta * G)

        print('生成返回值 m_gt')

        for i in range(G):
            if (group_topic_ind[i].sum() * group_doc_num[i] + alpha_class)!=0:
                m_gt_t[i] = (group_doc_num[i] * group_topic_ind[i] + alpha_class * mu) / (
                        group_topic_ind[i].sum() * group_doc_num[i] + alpha_class)
            else:
                m_gt_t[i] = mu

    pid = os.getpid()

    path_chain_model = osp.join(root, my_dataset,
                                   '{dataset}_{process_id}_model.npz'.format(dataset=my_dataset,
                                                                                 process_id=pid))  # 当前链的模型

    np.savez(path_chain_model, g_phi=g_phi, g_n_w_out=g_n_w_1.T, m_gt_t=m_gt_t, phi=phi, n_w_out=n_w_1.T, alpha_d=alpha_doc, alpha_g=alpha_group)

    print('训练模型存储', path_chain_model)




    return path_chain_model         #返回模型地址


'''
###############################################################################
                              LHLDA 模型训练
###############################################################################
'''

if __name__ == '__main__':
    if ((TrainTestOutput & 4) == 4):
        phi_all = np.zeros((K, V))
        n_w_out_all = np.zeros((K, V))

        g_phi_all = np.zeros((G, V))
        g_n_w_out_all = np.zeros((G, V))

        m_gt_all = np.zeros((G, K))
        train_para = np.zeros(2)

        pool = multiprocessing.Pool(processes=Training_Chain_Num)
        train_result = []

        for j in range(Training_Chain_Num):
            print('第%d个训练链启动' % (j + 1))
            train_result.append(pool.apply_async(LHLDA_Yahoo_train, args=(doc2topic_ind,
                                                                          train_id_words4doc,
                                                                          train_id_words4words,
                                                                          M, V, K, G, alpha_doc,
                                                                          alpha_class,
                                                                          alpha_group,
                                                                          g_alpha_class, mu,
                                                                          g_mu, beta, g_beta,
                                                                          max_iter, auto_gap,
                                                                          auto_pand,
                                                                          doc2NeedFlag,
                                                                          group_flag,
                                                                          group_doc_num,
                                                                          group_topic_ind,)))
        pool.close()
        pool.join()
        print('各链训练结束，即将存储模型')
        l = 1
        for res in train_result:
            ret = res.get()
            model_file = np.load(ret)


            phi_all += model_file['phi']
            n_w_out_all += model_file['n_w_out']

            train_para[0] += model_file['alpha_d']
            train_para[1] += model_file['alpha_g']


            print('第%d链phi获取' % (l))
            if UseGMode == 1:
                g_phi_all += model_file['g_phi']
                g_n_w_out_all += model_file['g_n_w_out']
                m_gt_all += model_file['m_gt_t']

                print('第%d链g获取' % (l))
            l += 1

        phi = phi_all / Training_Chain_Num
        n_w_out = n_w_out_all / Training_Chain_Num
        n_w_out.astype(int)
        train_p = train_para / Training_Chain_Num

        print('训练已完成')
        np.save(path_phi, phi)
        np.save(path_nw, n_w_out)
        np.save(path_para, train_p)
        if UseGMode == 1:
            g_phi=g_phi_all/Training_Chain_Num
            g_n_w_out=g_n_w_out_all/Training_Chain_Num
            m_gt=m_gt_all/Training_Chain_Num
            np.save(path_g_phi, g_phi)
            np.save(path_g_nw, g_n_w_out)
            np.save(path_m_gt, m_gt)
        print('模型存储成果')
        localtime = time.asctime(time.localtime(time.time()))
        print('训练结束时间为 :', localtime)

'''
###############################################################################
                     测试样本真实标签(示性矩阵) & 标签数量
###############################################################################
'''
if __name__ == '__main__':
    M_test_all = int(max(test_id_topic4doc_all)[0])  # 测试文本数量

    test_reallabel_ind = np.zeros((M_test_all, L), dtype='int')  # 真实标签（正确率指标计算的基础）
    for i in range(len(test_id_topic4doc_all)):
        test_reallabel_ind[test_id_topic4doc_all[i][0] - 1, test_id_topic4topic[i][0] - 1] = 1

    test_reallabel_num = np.zeros((M_test_all,), dtype='int')  # 标签数量
    for i in range(M_test_all):
        test_reallabel_num[i] = test_reallabel_ind[i, :].sum()

'''
###############################################################################
                           LHLDA 主程序 (test)
###############################################################################
'''
def LHLDA_Yahoo_test2(test_id_words4doc, test_id_words4words, phi, n_w_out, test_iter, Batch_Base,
                      alpha_doc_test):
    '''
    参数说明
    '''
    # test_id_words4doc：测试文本的文本索引
    # test_id_words4words：测试文本的词索引
    # phi：主题-词分布
    # n_w_out：主题-词频次统计
    # test_iter：测试迭代数

    ###########################################################################
    # 产生部分全局参数
    ###########################################################################

    import numpy as np
    M_test = int(max(test_id_words4doc)[0] - min(test_id_words4doc)[0] + 1)  # 测试文本数量
    K_test = n_w_out.shape[0]  # 主题数量
    V_test = n_w_out.shape[1]  # 语料库数量

    alpha_doc_test = 50  # 文章层alpha初始值
    # alpha_doc_test = alpha_doc  #不变
    m_gt_test = 1 / K_test * np.ones((K_test,))  # 文章层m_gt
    beta_test = 0.01 * np.ones((K_test,))

    # remark. 由于测试的时候没有标签信息，所以自动退为一层的LHLDA模型

    ###########################################################################
    # 构建数据存储数组（注意测试中不再是简单计数而改为了得分）
    ###########################################################################

    wordnum4doc = np.zeros((M_test, 1), dtype="int")  # 每个文档中词的总数
    for i in range(M_test):
        wordnum4doc[i] = np.sum(test_id_words4doc == [Batch_Base + i + 1])

    Avg_Doc_Word_Num = wordnum4doc.sum() / M_test  # 文档平均单词数

    Z_1_test = np.array([[0 for j in range(int(wordnum4doc[i]))] for i in range(M_test)])  # 所有文章所有词的主题

    n_d_1_test = np.zeros((M_test, K_test), dtype="int")  # 所有文档的主题得分分布
    n_d_1_sum_test = np.zeros(M_test)  # 所有文档的得分总和

    n_w_1_test = n_w_out.T.copy()
    n_w_1_sum_test = n_w_1_test.sum(axis=0)

    Phi_test = phi.T.copy()

    ###########################################################################
    # 初始化分配主题
    ###########################################################################

    for i in range(M_test):
        if i % 50 == 0:
            print('正在初始化第 %d 个测试样本' % (i))
        words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

        for j in range(int(wordnum4doc[i])):
            myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值

            P_1 = Phi_test[myword_ind]
            P_std_1 = np.squeeze(np.asarray(P_1 / P_1.sum()))  # 归一化

            topic_1 = np.argmax(np.random.multinomial(1, P_std_1))  # 产生一个主题

            Z_1_test[i][j] = topic_1

            n_d_1_test[i, topic_1] += 1
            n_d_1_sum_test[i] += 1

    ###########################################################################
    # 迭代
    ###########################################################################

    t_test = 1  # 迭代步数初始化
    while t_test <= test_iter:



        for i in range(M_test):
            words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

            for j in range(int(wordnum4doc[i])):

                topic_1_old = Z_1_test[i][j]  # 提取一级主题
                myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值

                if n_d_1_test[i, topic_1_old] >= 1:
                    n_d_1_test[i, topic_1_old] -= 1
                    n_d_1_sum_test[i] -= 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 过长文章不要影响单词分布

                    if n_w_1_test[myword_ind, topic_1_old] >= 1:
                        n_w_1_test[myword_ind, topic_1_old] -= 1
                        n_w_1_sum_test[topic_1_old] -= 1

                if UsePhi == 1:
                    P_1_test = Phi_test[myword_ind] * (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test)  # 词的主题分布概率
                else:
                    P_1_test = (n_w_1_test[myword_ind] + beta_test) / (n_w_1_sum_test + beta_test * V_test) * (
                            (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test))  # 词的主题分布概率

                P_1_std_test = np.squeeze(np.asarray(P_1_test / P_1_test.sum()))  # 归一化
                # --------------------------------------------------------------
                topic_1_new = np.argmax(np.random.multinomial(1, P_1_std_test))  # 产生一个新的主题

                # --------------------------------------------------------------
                Z_1_test[i][j] = topic_1_new  # 把新的 topic_1 赋值给 Z_1
                n_d_1_test[i, topic_1_new] += 1
                n_d_1_sum_test[i] += 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 避免过长文章的影响

                    n_w_1_test[myword_ind, topic_1_new] += 1
                    n_w_1_sum_test[topic_1_new] += 1

        #######################################################################
        # 中间输出结果
        #######################################################################
        if t_test % 10 == 0:  # 每10次输出一次中间结果
            theta_temp = np.zeros((M_test, K_test))
            Temp_One_Error = 0
            for i in range(M_test):
                theta_temp[i, :] = (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                        n_d_1_sum_test[i] + alpha_doc_test)

                sort_temp_1 = np.argsort(-theta_temp[i, :])  # 概率值从大到小排序，索引
                temp_top_topic_sy_single = sort_temp_1[0:1]  # 当前新闻的 top topic
                temp_NO1_topic_ind_single = np.zeros((K_test,), dtype='int')  # 预测的 No.1 主题
                temp_NO1_topic_ind_single[temp_top_topic_sy_single[0]] = 1  # 当前新闻示性矩阵

                OneRight = np.bitwise_and(temp_NO1_topic_ind_single, test_reallabel_ind[Batch_Base + i, :])
                if OneRight.sum() == 0:
                    Temp_One_Error += 1
            Temp_One = 100 * Temp_One_Error / M_test
            print('当前的One_Error', t_test, Temp_One)

        #######################################################################
        # 更新Phi
        #######################################################################
        if (t_test % phi_refresh == 0) and (UsePhi == 1):
            print('更新Phi')
            for i in range(V_test):
                Phi_test[i] = (n_w_1_test[i] + beta_test) / (n_w_1_sum_test + beta_test * V_test)

        t_test += 1

    ###########################################################################
    # 计算 文档-主题分布
    ###########################################################################

    theta_test = np.zeros((M_test, K_test))
    for i in range(M_test):
        theta_test[i, :] = (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (n_d_1_sum_test[i] + alpha_doc_test)

    n_w_out_1 = n_w_1_test.T.copy()
    for i in range(V_test):
        Phi_test[i] = (n_w_1_test[i] + beta_test) / (n_w_1_sum_test + beta_test * V_test)
    phi_1 = Phi_test.T.copy()

    ###########################################################################
    # 输出
    ###########################################################################

    '''
    输出说明
    '''
    #    theta_test_1：测试样本的 文本-主题分布
    #    Z_1_test：测试样本的词的主题
    #    phi : 更新后的Phi
    #    n_w_out : 更新后的n_w_out

    return phi_1, n_w_out_1, theta_test, Z_1_test


'''
###############################################################################
                           LHLDA 主程序2 (test3)  多次采样取平均
###############################################################################
'''


class test_ret:
    def __init__(self, K, V, M):
        self.phi_1 = np.zeros((K, V))
        self.n_w_out_1 = np.zeros((K, V))
        self.theta_test_out = np.zeros((M, K))


def LHLDA_Yahoo_test3(test_id_words4doc, test_id_words4words, phi, n_w_out, test_iter, Batch_Base, g_theta,
                      g_m_gt, alpha_doc_test, G_test, L, test_reallabel_ind):
    '''
    参数说明
    '''
    # test_id_words4doc：测试文本的文本索引
    # test_id_words4words：测试文本的词索引
    # phi：主题-词分布
    # n_w_out：主题-词频次统计
    # test_iter：测试迭代数
    # g_theta: 属于哪个组的判定
    # g_m_gt: 根据组选择m_g

    ###########################################################################
    # 产生部分全局参数
    ###########################################################################

    import numpy as np
    np.random.seed()
    M_test = int(max(test_id_words4doc)[0] - min(test_id_words4doc)[0] + 1)  # 测试文本数量
    K_test = n_w_out.shape[0]  # 主题数量
    V_test = n_w_out.shape[1]  # 语料库数量

    ret = test_ret(K_test, V_test, M_test)
    # alpha_doc_test = 50  # 文章层alpha初始值
    # alpha_doc_test = alpha_doc  #不变
    m_gt_test = 1 / K_test * np.ones((K_test,))  # 文章层m_gt
    beta_test = 0.01 * np.ones((K_test,))
    theta_test = np.zeros((Sample_Num, M_test, K_test))
    theta_entropy = np.zeros(Sample_Num)

    m_gt_doc = np.zeros((M_test, K_test))
    if UseGMode == 1:
        for i in range(M_test):

            if Gconv == 0:
                sort_g_theta = np.argsort(-g_theta[i, :])
                temp_group = sort_g_theta[0:1]
                m_gt_doc[i] = g_m_gt[temp_group[0]]

            # 融合m_gt信息
            elif Gconv == 1:  # 全融合
                m_gt_temp = np.zeros(K_test)
                for j in range(G_test):
                    m_gt_temp += g_theta[i, j] * g_m_gt[j]
                m_gt_doc[i] = np.squeeze(np.asarray(m_gt_temp / m_gt_temp.sum()))

            elif Gconv == 2:  # 融合大的
                sort_g_theta = np.argsort(-g_theta[i, :])  # 概率值从大到小排序，索引

                Delta_group_prep = np.zeros(G_test - 1)
                for j in range(G_test - 1):
                    Delta_group_prep[j] = g_theta[i, sort_g_theta[j]] - g_theta[i, sort_g_theta[j + 1]]

                group_num = np.argmax(Delta_group_prep) + 1
                # 当前新闻top个组别
                test_top_group = sort_g_theta[0:group_num]

                m_gt_temp = np.zeros(K_test)
                for j in range(group_num):
                    m_gt_temp += g_theta[i, test_top_group[j]] * g_m_gt[test_top_group[j]]
                m_gt_doc[i] = np.squeeze(np.asarray(m_gt_temp / m_gt_temp.sum()))  # 归一化

    # remark. 由于测试的时候没有标签信息，所以自动退为一层的LHLDA模型

    # alpha_doc_test = 50  # 文章层alpha初始值
    # m_gt_test = 1 / K_test  # 文章层m_gt

    ###########################################################################
    # 构建数据存储数组（注意测试中不再是简单计数而改为了得分）
    ###########################################################################

    wordnum4doc = np.zeros((M_test, 1), dtype="int")  # 每个文档中词的总数
    for i in range(M_test):
        wordnum4doc[i] = np.sum(test_id_words4doc == [Batch_Base + i + 1])

    Avg_Doc_Word_Num = wordnum4doc.sum() / M_test  # 文档平均单词数

    Z_1_test = np.array([[0 for j in range(int(wordnum4doc[i]))] for i in range(M_test)])  # 所有文章所有词的主题

    n_d_1_test = np.zeros((M_test, K_test), dtype="int")  # 所有文档的主题得分分布
    n_d_1_sum_test = np.zeros(M_test)  # 所有文档的得分总和

    n_w_1_test = n_w_out.T.copy()
    n_w_1_sum_test = n_w_1_test.sum(axis=0)

    Phi_test = phi.T.copy()

    ###########################################################################
    # 初始化分配主题
    ###########################################################################

    for i in range(M_test):
        if i % 50 == 0:
            print('正在初始化第 %d 个测试样本' % (i))

        if UseGMode == 1:  # G模式使用分组不同的m_gt
            m_gt_test = m_gt_doc[i]

        words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

        for j in range(int(wordnum4doc[i])):
            myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值
            if myword_ind >= V_test:
                #print('out of bound')
                continue

            if UsePhi == 1:
                P_1_test = Phi_test[myword_ind] * (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                        n_d_1_sum_test[i] + alpha_doc_test)  # 词的主题分布概率
            else:
                P_1_test = (n_w_1_test[myword_ind] + beta_test) / (n_w_1_sum_test + beta_test * V_test) * (
                        (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                        n_d_1_sum_test[i] + alpha_doc_test))  # 词的主题分布概率

            P_1_std_test = np.squeeze(np.asarray(P_1_test / P_1_test.sum()))  # 归一化
            # --------------------------------------------------------------
            topic_1 = np.argmax(np.random.multinomial(1, P_1_std_test))  # 产生一个新的主题

            Z_1_test[i][j] = topic_1

            n_d_1_test[i, topic_1] += 1
            n_d_1_sum_test[i] += 1
            n_w_1_test[myword_ind, topic_1] += 1
            n_w_1_sum_test[topic_1] += 1

    ###########################################################################
    # 迭代
    ###########################################################################

    t_test = 1  # 迭代步数初始化
    Sample_time = 0
    while t_test <= (test_iter + (Sample_Num - 1) * Sample_Gap):

        for i in range(M_test):
            if UseGMode == 1:  # G模式使用分组不同的m_gt
                m_gt_test = m_gt_doc[i]

            words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

            for j in range(int(wordnum4doc[i])):

                topic_1_old = Z_1_test[i][j]  # 提取一级主题
                myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值
                if myword_ind >= V_test:
                    #print('out of bound')
                    continue

                if n_d_1_test[i, topic_1_old] >= 1:
                    n_d_1_test[i, topic_1_old] -= 1
                    n_d_1_sum_test[i] -= 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 过长文章不要影响单词分布

                    if n_w_1_test[myword_ind, topic_1_old] >= 1:
                        n_w_1_test[myword_ind, topic_1_old] -= 1
                        n_w_1_sum_test[topic_1_old] -= 1

                if UsePhi == 1:
                    P_1_test = Phi_test[myword_ind] * (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test)  # 词的主题分布概率
                else:
                    P_1_test = (n_w_1_test[myword_ind] + beta_test) / (n_w_1_sum_test + beta_test * V_test) * (
                            (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test))  # 词的主题分布概率

                P_1_std_test = np.squeeze(np.asarray(P_1_test / P_1_test.sum()))  # 归一化
                # --------------------------------------------------------------
                topic_1_new = np.argmax(np.random.multinomial(1, P_1_std_test))  # 产生一个新的主题

                # --------------------------------------------------------------
                Z_1_test[i][j] = topic_1_new  # 把新的 topic_1 赋值给 Z_1
                n_d_1_test[i, topic_1_new] += 1
                n_d_1_sum_test[i] += 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 避免过长文章的影响

                    n_w_1_test[myword_ind, topic_1_new] += 1
                    n_w_1_sum_test[topic_1_new] += 1

        if ((t_test >= test_iter) and (t_test % Sample_Gap == 0)):  # 采样
            theta_temp = np.zeros((M_test, K_test))
            # 输出采样theta的熵
            temp_theta_entropy = 0
            #Temp_One_Error = 0
            for i in range(M_test):
                theta_temp[i, :] = (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                        n_d_1_sum_test[i] + alpha_doc_test)

            print('第%d次采样' % (Sample_time + 1))

            theta_test[Sample_time] = theta_temp  # 记录采样值

            Sample_time += 1

        #######################################################################
        # 更新Phi
        #######################################################################
        if (t_test % phi_refresh == 0) and (UsePhi == 1):
            print('更新Phi')
            for i in range(V_test):
                Phi_test[i] = (n_w_1_test[i] + beta_test) / (n_w_1_sum_test + beta_test * V_test)

        t_test += 1

    ###########################################################################
    # 计算 文档-主题分布
    ###########################################################################


    ret.theta_test_out = theta_test.sum(axis=0) / Sample_Num

    ret.n_w_out_1 = n_w_1_test.T.copy()
    for i in range(V_test):
        Phi_test[i] = (n_w_1_test[i] + beta_test) / (n_w_1_sum_test + beta_test * V_test)
    ret.phi_1 = Phi_test.T.copy()

    ###########################################################################
    # 输出
    ###########################################################################

    '''
    输出说明
    '''
    #    theta_test_1：测试样本的 文本-主题分布
    #    Z_1_test：测试样本的词的主题
    #    phi : 更新后的Phi
    #    n_w_out : 更新后的n_w_out

    return ret


'''
###############################################################################
                           LHLDA 主程序2 (test_group)  获取组别
###############################################################################
'''


class group_ret:
    def __init__(self, G, V, M):
        self.g_phi_1 = np.zeros((G, V))
        self.g_n_w_out_1 = np.zeros((G, V))
        self.g_theta_test = np.zeros((M, G))


def LHLDA_Yahoo_test3_group(test_id_words4doc, test_id_words4words, g_phi, g_n_w_out, test_iter, Batch_Base,
                            alpha_group_test):
    '''
    参数说明
    '''
    # test_id_words4doc：测试文本的文本索引
    # test_id_words4words：测试文本的词索引
    # g_phi：分组-词分布
    # n_w_out：分组-词频次统计
    # test_iter：测试迭代数

    ###########################################################################
    # 产生部分全局参数
    ###########################################################################

    import numpy as np

    np.random.seed()

    M_test = int(max(test_id_words4doc)[0] - min(test_id_words4doc)[0] + 1)  # 测试文本数量
    G_test = g_n_w_out.shape[0]  # 分组数量
    V_test = g_n_w_out.shape[1]  # 语料库数量

    alpha_doc_test = alpha_group_test  # 文章层alpha初始值
    # alpha_doc_test = alpha_doc  #不变
    m_gt_test = 1 / G_test * np.ones((G_test,))  # 文章层m_gt
    beta_test = 0.01 * np.ones((G_test,))
    theta_test = np.zeros((M_test, G_test))


    # remark. 由于测试的时候没有标签信息，所以自动退为一层的LHLDA模型

    ###########################################################################
    # 构建数据存储数组（注意测试中不再是简单计数而改为了得分）
    ###########################################################################

    wordnum4doc = np.zeros((M_test, 1), dtype="int")  # 每个文档中词的总数
    for i in range(M_test):
        wordnum4doc[i] = np.sum(test_id_words4doc == [Batch_Base + i + 1])

    Avg_Doc_Word_Num = wordnum4doc.sum() / M_test  # 文档平均单词数

    Z_1_test = np.array([[0 for j in range(int(wordnum4doc[i]))] for i in range(M_test)])  # 所有文章所有词的主题

    n_d_1_test = np.zeros((M_test, G_test), dtype="int")  # 所有文档的主题得分分布
    n_d_1_sum_test = np.zeros(M_test)  # 所有文档的得分总和

    n_w_1_test = g_n_w_out.T.copy()
    n_w_1_sum_test = n_w_1_test.sum(axis=0)

    Phi_test = g_phi.T.copy()

    ###########################################################################
    # 初始化分配主题
    ###########################################################################

    for i in range(M_test):
        if i % 50 == 0:
            print('正在初始化第 %d 个测试样本' % (i))
        words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

        for j in range(int(wordnum4doc[i])):
            myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值
            if myword_ind >= V_test:
                #print('out of bound')
                continue
            P_1 = Phi_test[myword_ind]
            P_std_1 = np.squeeze(np.asarray(P_1 / P_1.sum()))  # 归一化

            topic_1 = np.argmax(np.random.multinomial(1, P_std_1))  # 产生一个主题

            Z_1_test[i][j] = topic_1

            n_d_1_test[i, topic_1] += 1
            n_d_1_sum_test[i] += 1
            n_w_1_test[myword_ind, topic_1] += 1
            n_w_1_sum_test[topic_1] += 1

    ###########################################################################
    # 迭代
    ###########################################################################

    t_test = 1  # 迭代步数初始化
    while t_test <= (test_iter + (Sample_Num - 1) * Sample_Gap):
        #print('分组进行第 %d 次迭代t...' % (t_test))

        for i in range(M_test):
            words_begin_now = int(np.sum(wordnum4doc[0:i]))  # 当前文章词索引的起始位置

            for j in range(int(wordnum4doc[i])):

                topic_1_old = Z_1_test[i][j]  # 提取一级主题
                myword_ind = int(test_id_words4words[words_begin_now + j]) - 1  # 当前词的索引值
                if myword_ind >= V_test:
                    #print('out of bound')
                    continue

                if n_d_1_test[i, topic_1_old] >= 1:
                    n_d_1_test[i, topic_1_old] -= 1
                    n_d_1_sum_test[i] -= 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 过长文章不要影响单词分布

                    if n_w_1_test[myword_ind, topic_1_old] >= 1:
                        n_w_1_test[myword_ind, topic_1_old] -= 1
                        n_w_1_sum_test[topic_1_old] -= 1

                if UsePhi == 1:
                    P_1_test = Phi_test[myword_ind] * (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test)  # 词的主题分布概率
                else:
                    P_1_test = (n_w_1_test[myword_ind] + beta_test) / (n_w_1_sum_test + beta_test * V_test) * (
                            (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                            n_d_1_sum_test[i] + alpha_doc_test))  # 词的主题分布概率

                P_1_std_test = np.squeeze(np.asarray(P_1_test / P_1_test.sum()))  # 归一化
                # --------------------------------------------------------------
                topic_1_new = np.argmax(np.random.multinomial(1, P_1_std_test))  # 产生一个新的主题

                # --------------------------------------------------------------
                Z_1_test[i][j] = topic_1_new  # 把新的 topic_1 赋值给 Z_1
                n_d_1_test[i, topic_1_new] += 1
                n_d_1_sum_test[i] += 1

                if j < Avg_Doc_Word_Num * LongLimit:  # 避免过长文章的影响

                    n_w_1_test[myword_ind, topic_1_new] += 1
                    n_w_1_sum_test[topic_1_new] += 1

        #######################################################################
        # 采样
        #######################################################################
        if ((t_test >= test_iter) and (t_test % Sample_Gap == 0)):  # 采样
            theta_temp = np.zeros((M_test, G_test))
            print('获取分组采样，进行第 %d 次迭代t...' % (t_test))

            for i in range(M_test):
                theta_temp[i, :] = (n_d_1_test[i, :] + (alpha_doc_test * m_gt_test)) / (
                        n_d_1_sum_test[i] + alpha_doc_test)
            theta_test += theta_temp  # 记录采样值

        t_test += 1



    '''
    输出说明
    '''
    #    theta_test_1：测试样本的 文本-主题分布
    #    Z_1_test：测试样本的词的主题
    #    phi : 更新后的Phi
    #    g_n_w_out : 更新后的n_w_out

    return theta_test / Sample_Num


###############################################################################
# 如果要缩减测试数据集会执行此部分操作
###############################################################################
if __name__ == '__main__':
    if ((TrainTestOutput & 2) == 2):
        if num_want > 0:
            import numpy as np

            test_num = int(np.min((num_want, np.max(test_id_topic4doc_all))))  # 确认测试数目没有超限

            test_id_topic4doc_all = np.array([x for x in test_id_topic4doc_all if x <= test_num])
            test_id_topic4topic = test_id_topic4topic[0:len(test_id_topic4doc_all)]

            test_id_words4doc_all = np.array([x for x in test_id_words4doc_all if x <= test_num])
            test_id_words4words_all = test_id_words4words_all[0:len(test_id_words4doc_all)]
            print('完成测试集缩减,测试集被缩减到 %d 个测试样本' % (test_num))

###############################################################################
# 如果要缩减测试数据集会执行此部分操作
###############################################################################

'''
###############################################################################
                              LHLDA 测试(Yahoo)
###############################################################################
'''
if __name__ == '__main__':
    if ((TrainTestOutput & 2) == 2):
        # 读取模型参数
        import numpy as np

        localtime = time.asctime(time.localtime(time.time()))
        print('开始测试时间 :', localtime)
        phi = np.load(path_phi)
        n_w_out = np.load(path_nw)
        g_phi = np.load(path_g_phi)
        g_n_w_out = np.load(path_g_nw)
        g_m_gt = np.load(path_m_gt)
        print('获取模型参数')
        train_para = np.load(path_para)
        if use_train_para_doc == 1:
            alpha_doc_test = train_para[0]
            print('使用训练迭代的参数alpha_doc', alpha_doc_test)
        if use_train_para_group == 1:
            alpha_group_test = train_para[1]
            print('使用训练迭代的参数alpha_group', alpha_group_test)
        # 支持批次预测
        M_test_all = int(max(test_id_words4doc_all)[0])  # 测试文本数量
        K_test = n_w_out.shape[0]
        theta_test_all = np.zeros((M_test_all, K_test))
        test_num = int(np.ceil(M_test_all / Test_Batch))
        print('开始测试，测试文档数',M_test_all)

        if Batch_Mode == 1:
            doc_test_num = 0
            for i in range(test_num):
                # 产生本批次测试文本数量
                if (i + 1) * Test_Batch > M_test_all:
                    M_test = M_test_all - i * Test_Batch
                else:
                    M_test = Test_Batch
                print('第%d个批次预测: %d ~ %d' % ((i + 1), i * Test_Batch + 1, i * Test_Batch + M_test))

                test_id_words4doc = []
                test_id_words4words = []

                while (test_id_words4doc_all[doc_test_num] <= i * Test_Batch + M_test):
                    test_id_words4doc.append(test_id_words4doc_all[doc_test_num])
                    test_id_words4words.append(test_id_words4words_all[doc_test_num])
                    doc_test_num += 1
                    if doc_test_num >= len(test_id_words4doc_all):
                        break

                test_id_words4doc = np.array(test_id_words4doc)
                test_id_words4words = np.array(test_id_words4words)

                G_test = g_n_w_out.shape[0]
                g_theta_test = np.zeros((M_test, G_test))
                g_phi_all = np.zeros((G, V))
                g_n_w_all = np.zeros((G, V))
                if UseGMode == 1:  # 先产生分组，确定m
                    pool = multiprocessing.Pool(processes=MultiGTest)
                    group_result = []
                    for j in range(MultiGTest):
                        print('第%d个分组链启动' % (j + 1))
                        group_result.append(pool.apply_async(LHLDA_Yahoo_test3_group, args=(test_id_words4doc,
                                                                                            test_id_words4words, g_phi,
                                                                                            g_n_w_out, test_iter,
                                                                                            i * Test_Batch,
                                                                                            alpha_group_test,)))
                    pool.close()
                    pool.join()
                    print('分组链预测本批次结束，输出数据')
                    l = 1
                    for res in group_result:
                        ret = res.get()
                        g_theta_test += ret
                        #if Batch_Refresh == 1:
                        #    g_phi_all += ret.g_phi_1
                        #    g_n_w_all += ret.g_n_w_out
                        print('G模式第%d链测试数据获取' % (l))
                        l += 1

                    g_theta_test /= MultiGTest
                    if Batch_Refresh == 1:
                        g_phi_all /= MultiGTest
                        g_n_w_all /= MultiGTest

                if MultiChainTestMode != 1:
                    phi_new, n_w_out_new, theta_test, Z_1_test = LHLDA_Yahoo_test2(test_id_words4doc,
                                                                                   test_id_words4words,
                                                                                   phi, n_w_out, test_iter,
                                                                                   i * Test_Batch, alpha_doc_test)
                else:
                    theta_test = np.zeros((M_test, K_test))
                    phi_new = np.zeros((K, V))
                    n_w_out_newn = np.zeros((K, V))

                    pool = multiprocessing.Pool(processes=Test_Chain_Num
                                                )
                    test_result = []
                    for j in range(Test_Chain_Num):
                        print('第%d个测试链启动' % (j + 1))
                        test_result.append(pool.apply_async(LHLDA_Yahoo_test3, args=(test_id_words4doc,
                                                                                     test_id_words4words, phi,
                                                                                     n_w_out, test_iter,
                                                                                     i * Test_Batch, g_theta_test,
                                                                                     g_m_gt,
                                                                                     alpha_doc_test, G_test, L,
                                                                                     test_reallabel_ind,)))
                    pool.close()
                    pool.join()
                    print('本批次测试结束，准备取数据')
                    l = 1
                    for res in test_result:
                        ret = res.get()
                        theta_test += ret.theta_test_out
                        if Batch_Refresh == 1:
                            phi_new += ret.phi_1
                            n_w_out_new += ret.n_w_out_1
                        print('第%d链测试数据获取' % (l))
                        l += 1

                    theta_test /= Test_Chain_Num
                    if Batch_Refresh == 1:
                        phi_new /= Test_Chain_Num
                        n_w_out_new /= Test_Chain_Num

                if Batch_Refresh == 1:
                    phi = phi_new.copy()
                    n_w_out = n_w_out_new.copy()
                    if UseGMode == 1:
                        g_phi = g_phi_all.copy()
                        g_n_w_out = g_n_w_all.copy()

                for j in range(M_test):
                    theta_test_all[i * Test_Batch + j] = theta_test[j].copy()
                # 输出到目前为止的One Error,每500次输出
                M_temp = i * Test_Batch + M_test
                if M_temp % 1000 == 0:
                    Temp_One_Error = 0
                    for l in range(M_temp):
                        theta_temp = theta_test_all[l].copy()
                        if LT > 0:
                            for j in range(LT):
                                theta_temp[L - j - 1] = 0
                        sort_temp_1 = np.argsort(-theta_temp)  # 概率值从大到小排序，索引
                        temp_top_topic_sy_single = sort_temp_1[0:1]  # 当前新闻的 top topic
                        temp_NO1_topic_ind_single = np.zeros((K_test,), dtype='int')  # 预测的 No.1 主题
                        temp_NO1_topic_ind_single[temp_top_topic_sy_single[0]] = 1  # 当前新闻示性矩阵

                        OneRight = np.bitwise_and(temp_NO1_topic_ind_single, test_reallabel_ind[l, :])
                        if OneRight.sum() == 0:
                            Temp_One_Error += 1

                    Temp_One = 100 * Temp_One_Error / M_temp
                    print('已处理%d个文本，当前的One Error为%f' % (M_temp, Temp_One))

                print('本批次结束，输出theta')
                if M_temp % 2000 == 0:  # 太多数据时，暂存结果
                    M_num = np.zeros(1)
                    M_num[0] = M_temp
                    np.savez(path_theta_test, theta_test=theta_test_all, M_test=M_num)
                    localtime = time.asctime(time.localtime(time.time()))
                    print('阶段测试时间为 :', M_temp, localtime)
        else:

            phi, n_w_out, theta_test_all, Z_1_test = LHLDA_Yahoo_test2(test_id_words4doc_all,
                                                                       test_id_words4words_all,
                                                                       phi, n_w_out, test_iter, 0,
                                                                       alpha_doc_test)
        M_num = np.zeros(1)
        M_num[0] = M_test_all
        np.savez(path_theta_test, theta_test=theta_test_all, M_test=M_num)
        localtime = time.asctime(time.localtime(time.time()))
        print('测试结束时间为 :', localtime)
'''
###############################################################################
                           单条样本的主题预测函数
###############################################################################
'''


def LHLDA_TopTopic_get_single(theta_test_single, top_number_single, K_test, multi_test_mode, test_reallabel_ind):
    '''
    参数说明
    '''
    # theta_test_single：单个文本主题分布
    # top_number_single：单个文本需要输出的主题数目
    # K_test：主题总数

    import numpy as np

    sort_temp_1 = np.argsort(-theta_test_single)  # 概率值从大到小排序，索引

    if multi_test_mode!=2:
        Delta_Topic_prep = np.zeros(K_test)
        for i in range(K_test - 1):
            Delta_Topic_prep[i] = theta_test_single[sort_temp_1[i]] - theta_test_single[sort_temp_1[i + 1]]


    else:       #BEP
        test_top_topic_sy_single = sort_temp_1[0:K_test]
        test_top_topic_ind_single = np.zeros((K_test,), dtype='int')  # 预测主题的示性矩阵
        i = 1
        while (i<=K_test):
            for j in range(i):
                test_top_topic_ind_single[test_top_topic_sy_single[j]] = 1

            #计算精度和召回率

            inters = np.bitwise_and(test_top_topic_ind_single,test_reallabel_ind)
            Right_Num = inters.sum()
            Precision = Right_Num/(test_reallabel_ind.sum())
            Recall = Right_Num/(test_top_topic_ind_single.sum())
            if (Precision == Recall):
                break
            i += 1
        top_number_single = i
        if Precision==0:
            top_number_single+=1


    test_top_topic_sy_single = sort_temp_1[0:top_number_single]  # 当前新闻的 top 个 topic 的索引

    test_top_topic_ind_single = np.zeros((K_test,), dtype='int')  # 预测主题的示性矩阵
    for i in range(top_number_single):
        test_top_topic_ind_single[test_top_topic_sy_single[i]] = 1

    test_NO1_topic_ind_single = np.zeros((K_test,), dtype='int')  # 预测的 No.1 主题
    test_NO1_topic_ind_single[test_top_topic_sy_single[0]] = 1
    '''
    输出说明
    '''
    # test_top_topic_sy_single：单个文本的主题索引
    # test_top_topic_ind_single：单个文本的主题示性向量
    # test_NO1_topic_ind_single：单个文本的 No.1 主题

    return test_top_topic_sy_single, test_top_topic_ind_single, test_NO1_topic_ind_single

class eval_ret:
    def __init__(self, M_test):
        self.M = M_test
        self.Doc_Precision: ndarray = np.zeros(M_test)
        self.Doc_Recall: ndarray = np.zeros(M_test)
        self.Doc_FScore: ndarray = np.zeros(M_test)
        self.Doc_Rankloss: ndarray = np.zeros(M_test)
        self.Doc_Margin: ndarray = np.zeros(M_test)
        self.All_Right_Num = 0
        self.OneError_Num = 0
        self.is_error_num = 0


'''

###############################################################################
                             分段统计函数
###############################################################################
'''
def LHLDA_evaluation_multiprocess(test_top_topic_ind, test_NO1_topic_ind, test_reallabel_ind, M_test, K_test, theta_test_all):

    import numpy as np
    import os

    ret = eval_ret(int(M_test))
    pid = os.getpid()

    if SimpleStatistics == 1:
        print('不统计Margin和IsError，简化计算')

    for i in range(int(M_test)):
        if i%100==0:
            print('输出指标，已统计文档数：',pid,i)
        inters = np.bitwise_and(test_top_topic_ind[i, :], test_reallabel_ind[i, :])  # 取并集
        OneInter = np.bitwise_and(test_NO1_topic_ind[i, :], test_reallabel_ind[i, :])  # 只取top1的并集
        Right_Num = inters.sum()
        One_Num = OneInter.sum()
        if One_Num == 0:
            ret.OneError_Num += 1
        ret.Doc_Precision[i] = Right_Num / test_reallabel_ind.sum(axis=1)[i]
        ret.Doc_Recall[i] = Right_Num / test_top_topic_ind.sum(axis=1)[i]
        if Right_Num != 0:
            ret.Doc_FScore[i] = 2 * ret.Doc_Recall[i] * ret.Doc_Precision[i] / (ret.Doc_Recall[i] + ret.Doc_Precision[i])
        else:
            ret.Doc_FScore[i] = 0
        ret.All_Right_Num += Right_Num

        # 计算Ranking loss
        loss_num = 0
        irrelevant_topic = 0
        for j in range(K_test - LT):  # 排除预测的隐主题
            if test_reallabel_ind[i, j] != 1:  # 对所有不正确的的主题
                irrelevant_topic += 1
                for l in range(K_test - LT):
                    if ((test_reallabel_ind[i, l] == 1) and (
                            theta_test_all[i, j] > theta_test_all[i, l])):  # 不相关主题的概率反而大于相关主题的概率
                        loss_num += 1
        ret.Doc_Rankloss[i] = loss_num / (irrelevant_topic * (K_test - LT - irrelevant_topic))

        if SimpleStatistics != 1:
            # 计算Margin
            sort_temp_1 = np.argsort(-theta_test_all[i])  # 概率值从大到小排序，索引
            max_irre_index = -1
            min_re_index = -1
            for j in range(K_test - LT):
                if (test_reallabel_ind[i, sort_temp_1[j]] == 1):  # relevant topic
                    min_re_index = j  # 从大到小排名，取到最小的
                else:  # irrelevent topic
                    if max_irre_index == -1:
                        max_irre_index = j  # 第一次赋值,是最大的
            if (min_re_index != -1) and (max_irre_index != -1):
                ret.Doc_Margin[i] = abs(min_re_index - max_irre_index)

            # 计算Is Error
            if (max_irre_index != -1) and (max_irre_index < (K_test - LT - irrelevant_topic)):
                ret.is_error_num += 1
    print('统计进程结束',pid)
    return ret

'''

###############################################################################
                             多进程调用函数
###############################################################################
'''

def LHLDA_evaluation_set3(test_top_topic_ind, test_NO1_topic_ind, test_reallabel_ind, M_test, K_test, theta_test_all):

    import numpy as np
    Doc_Precision: ndarray = np.zeros(M_test)
    Doc_Recall: ndarray = np.zeros(M_test)
    Doc_FScore: ndarray = np.zeros(M_test)
    Doc_Rankloss: ndarray = np.zeros(M_test)
    Doc_Margin: ndarray = np.zeros(M_test)
    All_Right_Num = 0
    OneError_Num = 0
    is_error_num = 0

    Multi_process_eval = 8  #启动8个进程
    M_test_proc = np.ceil(M_test/Multi_process_eval)
    pool = multiprocessing.Pool(processes=Multi_process_eval)
    eval_result = []

    for i in range(Multi_process_eval):

        if (i+1)*M_test_proc > M_test:
            M_test_curr = M_test - i*M_test_proc
        else:
            M_test_curr = M_test_proc

        print('第%d个统计评估链启动,统计%d文档' % (i + 1, M_test_curr))

        test_top_topic_ind_curr = np.zeros((int(M_test_curr), K_test), dtype='int')
        test_NO1_topic_ind_curr = np.zeros((int(M_test_curr), K_test), dtype='int')
        test_reallabel_ind_curr = np.zeros((int(M_test_curr), K_test), dtype='int')
        theta_test_all_curr = np.zeros((int(M_test_curr), K_test))

        test_top_topic_ind_curr[:] = test_top_topic_ind[int(i*M_test_proc):int(i*M_test_proc+M_test_curr)]
        test_NO1_topic_ind_curr[:] = test_NO1_topic_ind[int(i*M_test_proc):int(i*M_test_proc+M_test_curr)]
        test_reallabel_ind_curr[:] = test_reallabel_ind[int(i*M_test_proc):int(i*M_test_proc+M_test_curr)]
        theta_test_all_curr[:] = theta_test_all[int(i*M_test_proc):int(i*M_test_proc+M_test_curr)]

        eval_result.append(pool.apply_async(LHLDA_evaluation_multiprocess,args=(test_top_topic_ind_curr, test_NO1_topic_ind_curr, test_reallabel_ind_curr, M_test_curr, K_test, theta_test_all_curr,)))

    pool.close()
    pool.join()
    print('各评估链结束')
    EvalBase = 0


    for res in eval_result:
        ret = res.get()
        Doc_Precision[EvalBase:EvalBase+ret.M] = ret.Doc_Precision[:]
        Doc_Recall[EvalBase:EvalBase+ret.M] = ret.Doc_Recall[:]
        Doc_FScore[EvalBase:EvalBase+ret.M] = ret.Doc_FScore[:]
        Doc_Rankloss[EvalBase:EvalBase+ret.M] = ret.Doc_Rankloss[:]
        Doc_Margin[EvalBase:EvalBase+ret.M] = ret.Doc_Margin[:]
        All_Right_Num += ret.All_Right_Num
        OneError_Num += ret.OneError_Num
        is_error_num += ret.is_error_num
        EvalBase += ret.M


    Doc_Precision_Macro = 100 * Doc_Precision.sum() / M_test
    Doc_Recall_Macro = 100 * Doc_Recall.sum() / M_test
    Doc_FScore_Macro = 100 * Doc_FScore.sum() / M_test

    Doc_Precision_Micro = All_Right_Num / test_reallabel_ind[0:M_test].sum()
    Doc_Recall_Micro = All_Right_Num / test_top_topic_ind.sum()

    Doc_FScore_Micro = 2 * Doc_Precision_Micro * Doc_Recall_Micro / (Doc_Precision_Micro + Doc_Recall_Micro)

    Doc_Precision_Micro *= 100
    Doc_Recall_Micro *= 100
    Doc_FScore_Micro *= 100

    Doc_OneError = 100 * OneError_Num / M_test
    Doc_Rankloss = 100 * Doc_Rankloss.sum() / M_test
    Doc_Margin_out = Doc_Margin.sum() / M_test
    Doc_IsError = 100 * is_error_num / M_test

    return Doc_Precision_Micro, Doc_Recall_Micro, Doc_FScore_Micro, Doc_Precision_Macro, Doc_Recall_Macro, Doc_FScore_Macro, Doc_OneError, Doc_Rankloss, Doc_Margin_out, Doc_IsError


'''

###############################################################################
                             预测评价指标函数
###############################################################################
'''


def LHLDA_evaluation_set2(test_top_topic_ind, test_NO1_topic_ind, test_reallabel_ind, M_test, K_test, theta_test_all):
    '''
    #参数说明
    '''
    # test_top_topic_ind：预测主题示性矩阵
    # test_NO1_topic_ind：第一主题示性矩阵
    # test_reallabel_ind：真实示性矩阵
    # M_test：测试文本数量
    # K_test：主题数量
    import numpy as np
    Doc_Precision: ndarray = np.zeros(M_test)
    Doc_Recall: ndarray = np.zeros(M_test)
    Doc_FScore: ndarray = np.zeros(M_test)
    Doc_Rankloss: ndarray = np.zeros(M_test)
    Doc_Margin: ndarray = np.zeros(M_test)
    All_Right_Num = 0
    OneError_Num = 0
    is_error_num = 0

    if SimpleStatistics == 1:
        print('不统计Margin和IsError，简化计算')

    for i in range(M_test):
        if i%100==0:
            print('输出指标，已统计文档数：',i)
        inters = np.bitwise_and(test_top_topic_ind[i, :], test_reallabel_ind[i, :])  # 取并集
        OneInter = np.bitwise_and(test_NO1_topic_ind[i, :], test_reallabel_ind[i, :])  # 只取top1的并集
        Right_Num = inters.sum()
        One_Num = OneInter.sum()
        if One_Num == 0:
            OneError_Num += 1
        Doc_Precision[i] = Right_Num / test_reallabel_ind.sum(axis=1)[i]
        Doc_Recall[i] = Right_Num / test_top_topic_ind.sum(axis=1)[i]
        if Right_Num != 0:
            Doc_FScore[i] = 2 * Doc_Recall[i] * Doc_Precision[i] / (Doc_Recall[i] + Doc_Precision[i])
        else:
            Doc_FScore[i] = 0
        All_Right_Num += Right_Num

        # 计算Ranking loss
        loss_num = 0
        irrelevant_topic = 0
        for j in range(K_test - LT):  # 排除预测的隐主题
            if test_reallabel_ind[i, j] != 1:  # 对所有不正确的的主题
                irrelevant_topic += 1
                for l in range(K_test - LT):
                    if ((test_reallabel_ind[i, l] == 1) and (
                            theta_test_all[i, j] > theta_test_all[i, l])):  # 不相关主题的概率反而大于相关主题的概率
                        loss_num += 1
        Doc_Rankloss[i] = loss_num / (irrelevant_topic * (K_test - LT - irrelevant_topic))

        if SimpleStatistics != 1:
            # 计算Margin
            sort_temp_1 = np.argsort(-theta_test_all[i])  # 概率值从大到小排序，索引
            max_irre_index = -1
            min_re_index = -1
            for j in range(K_test - LT):
                if (test_reallabel_ind[i, sort_temp_1[j]] == 1):  # relevant topic
                    min_re_index = j  # 从大到小排名，取到最小的
                else:  # irrelevent topic
                    if max_irre_index == -1:
                        max_irre_index = j  # 第一次赋值,是最大的
            if (min_re_index != -1) and (max_irre_index != -1):
                Doc_Margin[i] = abs(min_re_index - max_irre_index)

            # 计算Is Error
            if (max_irre_index != -1) and (max_irre_index < (K_test - LT - irrelevant_topic)):
                is_error_num += 1

    Doc_Precision_Macro = 100 * Doc_Precision.sum() / M_test
    Doc_Recall_Macro = 100 * Doc_Recall.sum() / M_test
    Doc_FScore_Macro = 100 * Doc_FScore.sum() / M_test

    Doc_Precision_Micro = All_Right_Num / test_reallabel_ind[0:M_test].sum()
    Doc_Recall_Micro = All_Right_Num / test_top_topic_ind.sum()

    Doc_FScore_Micro = 2 * Doc_Precision_Micro * Doc_Recall_Micro / (Doc_Precision_Micro + Doc_Recall_Micro)

    Doc_Precision_Micro *= 100
    Doc_Recall_Micro *= 100
    Doc_FScore_Micro *= 100

    Doc_OneError = 100 * OneError_Num / M_test
    Doc_Rankloss = 100 * Doc_Rankloss.sum() / M_test
    Doc_Margin_out = Doc_Margin.sum() / M_test
    Doc_IsError = 100 * is_error_num / M_test

    '''
    #输出说明
    '''

    # Doc_Precision_Macro：精确度-宏平均
    # Doc_Precision_Micro：精确度-微平均

    # Doc_Recall_Macro：召回率-宏平均
    # Doc_Recall_Micro：召回率-微平均

    # Doc_FScore_Macro：F-1-宏平均
    # Doc_FScore_Micro：F-1-宏平均

    # Doc_OneError: One Error
    # Doc_rankloss: Ranking Loss

    # Doc_Margin_out: Margin
    # Doc_IsError: Is Error

    return Doc_Precision_Micro, Doc_Recall_Micro, Doc_FScore_Micro, Doc_Precision_Macro, Doc_Recall_Macro, Doc_FScore_Macro, Doc_OneError, Doc_Rankloss, Doc_Margin_out, Doc_IsError


'''
###############################################################################
                             测试集预测评价
###############################################################################
'''
if __name__ == '__main__':

    localtime = time.asctime(time.localtime(time.time()))
    print('开始统计时间 :', localtime)
    if ((TrainTestOutput & 1) == 1):

        theta_test_all_file = np.load(path_theta_test)
        theta_test_all = theta_test_all_file['theta_test']
        M_test_temp = theta_test_all_file['M_test']
        M_test_all = theta_test_all.shape[0]
        if M_test_all > M_test_temp[0]:
            print('测试未正常完成', M_test_temp[0])
            M_test_all = int(M_test_temp[0])
        test_top_topic_sy = [None] * M_test_all  # 索引
        test_top_topic_ind = np.zeros((M_test_all, L), dtype='int')  # 所有预测主题的示性矩阵（正确率指标计算的基础）
        test_NO1_topic_ind = np.zeros((M_test_all, L), dtype='int')  # No.1预测主题的示性矩阵（正确率指标计算的基础）
        print('获取测试数据，开始统计，测试文档数：', M_test_temp[0])
        # 将其它隐主题去掉,不应作为预测值  LT
        if LT > 0:
            for i in range(LT):
                theta_test_all[:, L - i - 1] = 0

        for i in range(M_test_all):
            if i % 200 == 0:
                print('预分析测试数据：', i)

            if multi_test_mode == 1:
                test_top_topic_sy[i], test_top_topic_ind[i, :], test_NO1_topic_ind[i, :] = LHLDA_TopTopic_get_single(
                    theta_test_all[i, :], test_reallabel_num[i], L, multi_test_mode,test_reallabel_ind[i])
            else:
                test_top_topic_sy[i], test_top_topic_ind[i, :], test_NO1_topic_ind[i, :] = LHLDA_TopTopic_get_single(
                    theta_test_all[i, :], Med_Topic_num, L, multi_test_mode,test_reallabel_ind[i])

        print('准备输出统计结果')
        if L>200:
            LHLDA_Doc_Precision_Micro, LHLDA_Doc_Recall_Micro, LHLDA_Doc_FScore_Micro, LHLDA_Doc_Precision_Macro, LHLDA_Doc_Recall_Macro, LHLDA_Doc_FScore_Macro, LHLDA_Doc_OneError, Doc_Rankloss, Doc_Margin, Doc_IsError = LHLDA_evaluation_set3(
                test_top_topic_ind, test_NO1_topic_ind, test_reallabel_ind, M_test_all, L, theta_test_all)
        else:
            LHLDA_Doc_Precision_Micro, LHLDA_Doc_Recall_Micro, LHLDA_Doc_FScore_Micro, LHLDA_Doc_Precision_Macro, LHLDA_Doc_Recall_Macro, LHLDA_Doc_FScore_Macro, LHLDA_Doc_OneError, Doc_Rankloss, Doc_Margin, Doc_IsError = LHLDA_evaluation_set2(
                test_top_topic_ind, test_NO1_topic_ind, test_reallabel_ind, M_test_all, L, theta_test_all)
        print('LHLDA One Error:', LHLDA_Doc_OneError)
        print('LHLDA Is Error:', Doc_IsError)
        print('LHLDA Ranking Loss:', Doc_Rankloss)
        print('LHLDA Margin:', Doc_Margin)
        if multi_test_mode == 1:
            print('LHLDA Calibrated Micro Precision:', LHLDA_Doc_Precision_Micro)
            print('LHLDA Calibrated Micro Recall:', LHLDA_Doc_Recall_Micro)
            print('LHLDA Calibrated Micro F_1:', LHLDA_Doc_FScore_Micro)
            print('LHLDA Calibrated Macro Precision:', LHLDA_Doc_Precision_Macro)
            print('LHLDA Calibrated Macro Recall:', LHLDA_Doc_Recall_Macro)
            print('LHLDA Calibrated Macro F_1:', LHLDA_Doc_FScore_Macro)
        elif multi_test_mode == 0:
            print('LHLDA Proportional Micro Precision:', LHLDA_Doc_Precision_Micro)
            print('LHLDA Proportional Micro Recall:', LHLDA_Doc_Recall_Micro)
            print('LHLDA Proportional Micro F_1:', LHLDA_Doc_FScore_Micro)
            print('LHLDA Proportional Macro Precision:', LHLDA_Doc_Precision_Macro)
            print('LHLDA Proportional Macro Recall:', LHLDA_Doc_Recall_Macro)
            print('LHLDA Proportional Macro F_1:', LHLDA_Doc_FScore_Macro)
        else:
            print('LHLDA BEP Micro Precision:', LHLDA_Doc_Precision_Micro)
            print('LHLDA BEP Micro Recall:', LHLDA_Doc_Recall_Micro)
            print('LHLDA BEP Micro F_1:', LHLDA_Doc_FScore_Micro)
            print('LHLDA BEP Macro Precision:', LHLDA_Doc_Precision_Macro)
            print('LHLDA BEP Macro Recall:', LHLDA_Doc_Recall_Macro)
            print('LHLDA BEP Macro F_1:', LHLDA_Doc_FScore_Macro)

        localtime = time.asctime(time.localtime(time.time()))
        print('统计结束时间为 :', localtime)
