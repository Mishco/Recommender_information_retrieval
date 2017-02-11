from datetime import time

import numpy as np
import pandas as pd
import sys
from sklearn import cross_validation
from sklearn import svm
import datetime

# Import matplotlib
import matplotlib.pyplot as plt

class dataSet:

    def __init__(self, name):
        self.name = name
        self.data = []
        self.header = []        # hlavicka pre jednotlive subory
        self.headerFill = []        # vyplnena hlavicka

    def add(self, x):
        self.data.append(x)

    def addheader(self, h):
        self.header.append(h)

    def addfillheader(self,x):
        self.headerFill.append(x)

    def getheader(self,i):
        return self.headerFill.pop(i)

# Predict the unknown ratings through the dot product of the latent features for users and items
def prediction(P,Q):
    return np.dot(P.T, Q)


def summary(name_of_file, header_file):
    obj_d = dataSet(name_of_file)

    train_activity_header = header_file
    obj_d.addheader(train_activity_header)

    print('---------------')
    print("Summary of file: {file}".format(file=name_of_file))
    print('---------------')
    _train_activity = pd.read_csv(name_of_file, sep=',', names=train_activity_header)
    obj_d.add(_train_activity)

    n_count = [None] * 50

    for i, name in enumerate(header_file):
        #tmp = str_to_class(name)
        code = "_train_activity.{tmp}.unique().shape[0]".format(tmp=name)
        n_count[0] = eval(code)
        obj_d.addfillheader(n_count[0])
        print("Number of unique {iteration} is {count}".format(iteration=name, count=n_count[0]))


    return obj_d

def main():
    #obj_train_data = summary('train_activity.csv', ['id', 'user_id', 'dealitem_id', 'deal_id', 'quantity', 'market_price', 'create_time'])
    #obj_train_data_dealItems = summary('train_dealitems.csv', ['id', 'deal_id', 'title_dealitem', 'coupon_text1', 'coupon_text2', 'coupon_begin_time', 'coupon_end_time'])
    #obj_test_data = summary('test_activity.csv',['id', 'user_id', 'dealitem_id', 'deal_id', 'quantity', 'market_price', 'create_time'])

    #train_data = pandas.DataFrame(obj_train_data.data)
    #test_data = pandas.DataFrame(obj_test_data.data)

    #n_users = obj_train_data.getheader(1)
    #n_items = obj_train_data.getheader(1)

    # Reading users file:

    # deal-id - konkretna zlava
    # dealitem-id rozne typy ten konkretne zlavy

    dheader_new = {'id', 'user_id', 'dealitem_id', 'deal_id', 'quantity', 'market_price', 'create_time'}
    dheader = {"id":int, "user_id": int, "dealitem_id": int, 'deal_id': int, "quantity":int, "market_price":float, "create_time":datetime}
    #users = pd.read_csv('train_activity.csv', sep=',', names=dheader, encoding='latin-1')
    users = pd.read_csv('train_activity.csv')


    # Reading ratings file:
    dheader = {"id", "deal_id", "title_dealitem", "coupon_text1","coupon_test2", "coupon_begin_time", "coupon_end_time"}
    items = pd.read_csv('train_dealitems.csv', sep=',', names=dheader, encoding='utf-8')

    #print (users.shape)
    #users.head()

    # print(items.shape)
    # items.head()

    from itertools import groupby
    #import pandas as pd
    #
    # df = pd.DataFrame([['A', 'C', 'A', 'B', 'C', 'A', 'B', 'B', 'A', 'A'],
    #                    ['ONE', 'TWO', 'ONE', 'ONE', 'ONE', 'TWO', 'ONE', 'TWO', 'ONE', 'THREE']]).T
    # df.columns = [['Alphabet', 'Words']]
    # print(df)

    # print(users['dealitem_id'])








    #sortedreader = sorted(users, key=lambda d: (d['id'], d['deal_id']))

    #roups = groupby(sortedreader, key=lambda d: (d['id'], d['deal_id']))

    #
    # grouped = users.groupby(users['id']).agg(lambda x: ','.join(x))#['deal_id']
    #
    # for id in grouped.index:
    #     #print("{id} a {deal}".format(id=id, deal=deal_id))
    #     print("id = {}, deal = {}".format(id, users.loc[id]))
    #


    groups = users.groupby(users['id'])
    print(groups)


#       print(users.groupby(users['id'])['deal_id'])

#
#     things = [("animal", "bear"), ("animal", "duck"), ("plant", "cactus"), ("vehicle", "speed boat"),
#               ("vehicle", "school bus")]
#
#     for key, group in groupby(things, lambda x: x[0]):
#         for thing in group:
#             #print ("A %d is %d." % (key, thing[1]))
#             print("ID: " + key + " deal:" + thing[1])
#         print (" ")
#
# #
    # for u in users.id:
    #     df1 = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])
    #     Frame = Frame.append(pd.DataFrame(data=u), ignore_index=True)




    return

    #_train_activity = pd.read_csv('train_activity.csv', sep=',' , names=dheader) #names=['id', 'user_id', 'dealitem_id', 'deal_id', 'quantity', 'market_price', 'create_time'])
    #_test_activity = pd.read_csv('test_activity.csv', sep=',' , names=dheader) #names=['id', 'user_id', 'dealitem_id', 'deal_id', 'quantity', 'market_price', 'create_time'])

    # n_users = _train_activity.id.unique().shape[0]
    # n_items = _test_activity.id.unique().shape[0]
    #
    # print("Number of users: " + str(n_users))
    # print("Number of items: " + str(n_items))

    # trainSplitData, trainOtherData = cross_validation.train_test_split(_train_activity, test_size=0.25)
    # testSplitData, testOtherData = cross_validation.train_test_split(_test_activity, test_size=0.25)

    n_users = 100#len(trainOtherData)#10000#trainOtherData.id.unique().shape[0]
    n_items = 100#len(testOtherData)#10000#testOtherData.id.unique().shape[0]

    plt.hist(_train_activity["dealitem_id"])
    plt.show()

    print("Number of users: " + str(n_users))
    print("Number of items: " + str(n_items))

    # Create training and test matrix
    R = np.zeros((n_users, n_items))
    for line in trainOtherData.itertuples():
        R[int(line[1])-1, int(line[2])-1] = line[3]

    T = np.zeros((n_users, n_items))
    for line in testOtherData.itertuples():
        T[int(line[1])-1, int(line[2])-1] = line[3]

    lmbda = 0.1  # Regularisation weight
    k = 20  # Dimensionality of the latent feature space
    m, n = R.shape  # Number of users and items
    n_epochs = 100  # Number of epochs
    gamma = 0.01  # Learning rate

    P = 3 * np.random.rand(k, m)  # Latent user feature matrix
    Q = 3 * np.random.rand(k, n)  # Latent movie feature matrix


    # Zmensit si maticu

    # pri useroch ma zaujima deal
    #
    # vysku znizit pouzivatelov podla toho ak maju iba jeden alebo dva kupene itemy
    # tak neviem presne urcit ak su si podobny

    # sirku obmedzit klucovymi slovami , najfrekventovanejsie slova
    # sirka sa da obmedzit aj odstranim nul (indexi v elasticu)
    #

    # zeefektivnit vypocitavanie podobnosti (napriklad pri cosinusovej podobnosti)
    #   nemusime pocitat menovatel pre kazdeho pouzivatela vzdy nanovo zvlast
    #   ale vieme si ich predpocitat a potom pracovat uz s konstantami

    ###################################
    # Tri pripady rozdelenie ludi podla
    #

    # Viac ako 1, 2 veci
    # deal

    # klucove slova = najfrekventovanejsie slova - sirka
    # vyska

    # podobnost vektorov - cosinousov podobnost
    # prepdpocitat menovatela pre user's

    # vzdy vypocitat presnost pri kazdej mnozine

    # User-Item Collaborative Filtering
    # 1. Ci si nieco kupili, aspon jeden item
    #
    # Item-Item filtering
    # 2. V pripade ze nic nekupili
    # 3. Alebo ponuknutie akciovych(t.z.: rychlo konciacich itemov par hodin/dni a z tych cenovo najlacnejsich)
    #
    #

    # cosinovy
    # jagardova podobnost
    # podobnost

    # X = [[0, 0], [1, 1]]
    # y = [0, 1]
    # clf = svm.SVC()
    # clf.fit(X, y)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

if __name__ == '__main__':
    main()
