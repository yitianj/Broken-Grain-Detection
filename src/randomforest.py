import math
import sys
from IPython import embed
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
## dict: key is line index, value is "label attr0:value attr1:value.."
#train_data = {}  #{0: [], 1: []} # [[], []]
#test_data = {}

        #test_label_.append(train_label[idx])
#train_label_ = train_label[all_node_index[:num_train]]
#test_ = train_data[all_node_index[num_train:]]
#test_label_ = train_label[all_node_index[num_train:]]
# embed()
global train_data, train_label

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
            tmp = []
            for k,v in enumerate(line.strip().split(",")[1:]):
                tmp.append(float(v))
            train_data.append(tmp)
            train_label.append(1)
        for line in NIN:
            tmp = []
            for k,v in enumerate(line.strip().split(",")[1:]):
                tmp.append(float(v))
            train_data.append(tmp)
            train_label.append(0)

    train_data = np.array(train_data)
    train_label = np.array(train_label)

    all_node_index = list(range(len(train_data)))
    num_train = int(0.8 * len(train_data))
    for run in range(100):
        np.random.shuffle(all_node_index)
        train_, test_, train_label_, test_label_ = train_data[all_node_index[:num_train]], train_data[all_node_index[num_train:]], train_label[all_node_index[:num_train]], train_label[all_node_index[num_train:]]
        
        rf = RandomForestClassifier(max_depth=11, random_state=0)
        rf.fit(train_, train_label_)
        predict_train = rf.predict(train_)
        # embed()
        avg_train_acc.append( (predict_train==train_label_).sum() / len(train_label_))
        # rf.fit(test_, train_label_)
        predict_test = rf.predict(test_)
        avg_test_acc.append( (predict_test==test_label_).sum() / len(test_label_))

       
        p,r,f,s = precision_recall_fscore_support(predict_test, test_label_, average='macro')
        # embed()
        avg_test_f1.append(f)
        avg_test_rec.append(r)
        avg_test_prec.append(p)
print(np.mean(avg_train_acc))
print(np.mean(avg_test_acc))
print(np.mean(avg_test_f1))
print(np.mean(avg_test_rec))
print(np.mean(avg_test_prec))
#print()
#embed()