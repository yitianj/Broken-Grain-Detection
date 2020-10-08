import math
import sys
from IPython import embed
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
## dict: key is line index, value is "label attr0:value attr1:value.."
#train_data = {}  #{0: [], 1: []} # [[], []]
#test_data = {}

        #test_label_.append(train_label[idx])
#train_label_ = train_label[all_node_index[:num_train]]
#test_ = train_data[all_node_index[num_train:]]
#test_label_ = train_label[all_node_index[num_train:]]
# embed()
global train_data, train_label
if False:
    while True:
        try:
            s=input()
            label = s.split(" ")[0]
            if (label != "0"): 
                # train_data[idx] = s.split(" ")
                # tmp = [{x.split(':')[0]: x.split(':')[1]} ]
                tmp = {}
                for x in s.split(" ")[1:]:
                    tmp[x.split(':')[0]] = float(x.split(':')[1])
                attr_list |= set(tmp.keys())
                train_data.append(tmp)
                train_label.append(label)
                idx += 1
            else :
                #test_data[idx-len(train_data)] = s.split(" ")
                tmp = {}
                for x in s.split(" ")[1:]:
                    tmp[x.split(':')[0]] = float(x.split(':')[1])
                test_data.append(tmp)
                idx += 1           
        except EOFError as error:
            break

def get_gini(less_than_idx, greater_than_idx) :
    coef1 = float(len(less_than_idx))/(len(less_than_idx)+len(greater_than_idx))
    coef2 = float(len(greater_than_idx))/(len(less_than_idx)+len(greater_than_idx))
    gini1 = 1
    gini2 = 1
    temp_labels = {}
    for idx in less_than_idx:
        label_ = train_label[idx] 
        if label_ not in temp_labels: temp_labels[label_] = 1
        else: temp_labels[label_] += 1
    for k,v in temp_labels.items() :
        gini1 -= ((float(v)/len(less_than_idx))**2)

    temp_labels1 = {}
    for idx in greater_than_idx :
        label_ = train_label[idx] 
        if label_ not in temp_labels1: temp_labels1[label_] = 1
        else: temp_labels1[label_] += 1
    for k,v in temp_labels1.items() :
        gini2 -=  ((float(v)/len(greater_than_idx))**2)

    return (coef1*gini1 + coef2*gini2)

from collections import defaultdict


class DecisionTreeNode:

    def __init__(self, data_idx, the_depth, attr_list, max_depth = 3):
        self.left = None
        self.right = None
        self.data = data_idx
        self.attr_list = attr_list
        self.num_attrs = len(attr_list)
        self.split_attr = None
        self.split_value = None
        self.label = None
        #self.attr_idx = 1000  #default value: when node is leaf 
        #self.threshold = 1000 #default value: when node is leaf
        self.depth = the_depth
        self.max_depth = max_depth
        
    def train(self):
        if self.depth == self.max_depth:
            # print("dafsdf")
            vote = defaultdict(int) 
            for idx in self.data:
                vote[train_label[idx]] += 1
            largest = max(vote.values())
            for i in sorted(vote.keys()):
                if vote[i] == largest:
                    self.label = i
                    break
            # print("label", self.label)
            return
        min_gini = 1e10
        best_left_idx = []
        best_right_idx = []
        for attr_idx in range(self.num_attrs) :
            unique_values = set()

            ## find possible splits
            for idx in self.data:
            # for k,v in data.items() :
                # = float(v[i].split(":")[1])
                #values[k] = a 
                unique_values.add(train_data[idx][self.attr_list[attr_idx]])
            #print(unique_values)
            #continue
            unique_values = sorted(list(set(unique_values)))
            if len(unique_values) == 1 : continue
            j = 0
            possible_splits = []
            #
            while j+1 < len(unique_values):
                try:
                    split_ = float(unique_values[j]+unique_values[j+1])/2
                except:
                    embed()
                possible_splits.append(split_)
                j += 1
            
            ## min_gin, threshold
            for split_ in possible_splits :
                less_than_idx = [idx for idx in self.data if train_data[idx][self.attr_list[attr_idx]] < split_] 
                greater_than_idx = [idx for idx in self.data if train_data[idx][self.attr_list[attr_idx]] >= split_]
                gini_ = get_gini(less_than_idx, greater_than_idx)
                if gini_ < min_gini :
                    min_gini = gini_
                    # threshold = split_
                    best_left_idx = less_than_idx
                    best_right_idx = greater_than_idx
                    self.split_attr = self.attr_list[attr_idx]
                    self.split_value = split_
            # attribute_candidates[i] = [min_gini, best_left_idx, best_right_idx, threshold]
                # print(gini_, split_)
        if len(best_right_idx) == 0 or len(best_left_idx) == 0:
            vote = defaultdict(int) 
            for idx in self.data:
                vote[train_label[idx]] += 1
            largest = max(vote.values())
            for i in sorted(vote.keys()):
                if vote[i] == largest:
                    self.label = i
                    break
            #self.label = 
        else:
            self.left = DecisionTreeNode(best_left_idx, self.depth+1, self.attr_list)
            self.left.train()
            self.right = DecisionTreeNode(best_right_idx, self.depth+1, self.attr_list)
            self.right.train()
        return 
        

    def predict(self, data):
        result = self.label
        while result is None:
            if data[self.split_attr] < self.split_value:
                result = self.left.predict(data)
            else:
                result = self.right.predict(data)
        return result

    def _print(self):
        print("current depth:{}".format(self.depth))
        print(self.split_attr, self.split_value)
        if self.left:
            print("left")
            self.left._print()
        if self.right:
            print("right")
            self.right._print()
        if self.right is None and self.right is None:
            print(self.label)

if __name__ == "__main__":

    train_data, train_label, test_data = [], [], []
    avg_train_acc, avg_test_acc = [], []
    avg_test_f1, avg_test_prec, avg_test_rec = [], [], []
    attr_list = set()
    idx = 0

    # train_idx = 
    positive_file = sys.argv[1]
    negative_file = sys.argv[2]

    with open (positive_file, 'r') as PIN, open(negative_file, 'r') as NIN:
        attr_list = PIN.readline().strip().split(',')[1:]
        # embed()
        
        NIN.readline()
        for line in PIN:
            tmp = {}
            for k,v in enumerate(line.strip().split(",")[1:]):
                tmp[attr_list[k]] = float(v)
            train_data.append(tmp)
            train_label.append(1)
        for line in NIN:
            tmp = {}
            for k,v in enumerate(line.strip().split(",")[1:]):
                tmp[attr_list[k]] = float(v)
            train_data.append(tmp)
            train_label.append(0)


    all_node_index = list(range(len(train_data)))
    num_train = int(0.8 * len(train_data))
    for run in range(100):
        np.random.shuffle(all_node_index)
        train_, test_, train_label_, test_label_ = [], [], [], []
        for _, idx in enumerate(all_node_index):
            if _ <= num_train:
                train_.append(idx)
                #train_label_.append(train_label[idx])
            else:
                test_.append(idx)


        DT = DecisionTreeNode(train_, 0, attr_list)
        DT.train()

        cnt = 0
        for idx in train_:
            label = DT.predict(train_data[idx])
            if label == train_label[idx]:
                cnt += 1
        avg_train_acc.append(float(cnt) / len(train_))

        cnt = 0
        prediction, gt = [], []
        for idx in test_:
            label = DT.predict(train_data[idx])
            prediction.append(label)
            gt.append(train_label[idx])
            if label == train_label[idx]:
                cnt += 1
        avg_test_acc.append(float(cnt) / len(test_))
        p,r,f,s = precision_recall_fscore_support(prediction, gt, average='macro')
        # embed()
        avg_test_f1.append(f)
        avg_test_rec.append(r)
        avg_test_prec.append(p)
print(np.mean(avg_train_acc + avg_test_acc))
print(np.mean(avg_test_acc))
print(np.mean(avg_train_acc))
