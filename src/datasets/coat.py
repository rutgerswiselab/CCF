# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
import json
import random
import time
from datetime import datetime
import pdb


np.random.seed(DEFAULT_SEED)
host_name = socket.gethostname()
print(host_name)


RAW_DATA = PRE_DATA_DIR + 'Coat/'

def strTimeProp(start, end, prop, frmt):
    stime = time.mktime(time.strptime(start, frmt))
    etime = time.mktime(time.strptime(end, frmt))
    ptime = stime + prop * (etime - stime)
    return int(ptime)
 
def randomTimestamp(start, end, frmt='%Y-%m-%d %H:%M:%S'):
    return strTimeProp(start, end, random.random(), frmt)

def assign_time(infile, out_csv, start, end):
    f = open(infile, 'r')
    matrix = []
    for line in f:
        matrix.append([int(i) for i in line.split()])
    matrix = np.array(matrix)
    triples = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > 0:
                triples.append([i+1, j+1, matrix[i][j]])
    out_df = pd.DataFrame()
    out_df[UID] = [row[0] for row in triples]
    out_df[IID] = [row[1] for row in triples]
    out_df[LABEL] = [row[2] for row in triples]
    out_df[TIME] = [randomTimestamp(start, end) for row in triples]
    
    out_df = out_df.sort_values(by=[TIME, UID, IID])
    out_df = out_df.drop_duplicates([UID, IID]).reset_index(drop=True)

    out_df.to_csv(out_csv, sep='\t', index=False)
    # print(out_df)
    return out_df
    
def labelRating(infile, out_csv):
    data = pd.read_csv(infile, sep=SEP)
    data[LABEL] = data[LABEL].apply(lambda x: 1 if x > 3 else 0)
    data.to_csv(out_csv, sep='\t', index=False)
    
def divide_data(train_csv, test_csv, dataset_name, leave_n=1, warm_n=5):
    dir_name = os.path.join(PRE_DATASET_DIR, dataset_name)
    if not os.path.exists(dir_name): 
        os.mkdir(dir_name)
    train_data = pd.read_csv(train_csv, sep=SEP)
    train_set, split_set = leave_out_by_time_df(
        train_data, warm_n=warm_n, leave_n=leave_n, split_n=1, max_user=MAX_VT_USER)
    validation_set = split_set[0]
    test_set = pd.read_csv(test_csv, sep=SEP)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))
    if UID in train_set.columns:
        print('train_user=%d validation_user=%d test_user=%d' %
              (len(train_set[UID].unique()), len(validation_set[UID].unique()), len(test_set[UID].unique())))

    train_set.to_csv(os.path.join(dir_name, dataset_name + TRAIN_SUFFIX), index=False, sep=SEP)
    validation_set.to_csv(os.path.join(dir_name, dataset_name + VALIDATION_SUFFIX), index=False, sep=SEP)
    test_set.to_csv(os.path.join(dir_name, dataset_name + TEST_SUFFIX), index=False, sep=SEP)
    
    return train_set, validation_set, test_set
    
    
def main():
    train_start = '2018-06-01 00:00:00'
    train_end = '2019-01-01 00:00:00'
    test_start = '2019-01-01 00:00:01'
    test_end = '2019-05-01 00:00:00'
    train_input_path = RAW_DATA + 'train.ascii'
    test_input_path = RAW_DATA + 'test.ascii'
    train_csv = RAW_DATA + 'train.csv'
    test_csv = RAW_DATA + '.test.csv'
    train01_csv = RAW_DATA + 'train01.csv'
    test01_csv = RAW_DATA + '.test01.csv'
    assign_time(train_input_path, train_csv, train_start, train_end)
    assign_time(test_input_path, test_csv, test_start, test_end)
    labelRating(train_csv, train01_csv)
    labelRating(test_csv, test01_csv)
    dataset_name = 'Coat'
    divide_data(train01_csv, test01_csv, dataset_name)
    
if __name__ == '__main__':
    main()
