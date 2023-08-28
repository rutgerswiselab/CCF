# coding=utf-8

# default paras
DEFAULT_SEED = 2018
SEP = '\t'
SEQ_SEP = ','
MAX_VT_USER = 100000  # When apply leave out by time policy，the maximum number of users

# Path
PRE_DATA_DIR = '../../data/'
PRE_DATASET_DIR = '../../dataset/'
DATA_DIR = '../data/' 
DATASET_DIR = '../dataset/' 
MODEL_DIR = '../model/' 
LOG_DIR = '../log/'  # log path
RESULT_DIR = '../result/'  # results path
COMMAND_DIR = '../command/'  
LOG_CSV_DIR = '../log_csv/' 


# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.csv'  # suffix of training file
VALIDATION_SUFFIX = '.validation.csv'  # suffix of validation file
TEST_SUFFIX = '.test.csv'  # suffix of testing file
INFO_SUFFIX = '.info.json'  # suffix of dataset information file
USER_SUFFIX = '.user.csv'  # suffix of user feature file
ITEM_SUFFIX = '.item.csv'  # suffix of item feature file
TRAIN_POS_SUFFIX = '.train_pos.csv'  
VALIDATION_POS_SUFFIX = '.validation_pos.csv'  
TEST_POS_SUFFIX = '.test_pos.csv'  
TRAIN_NEG_SUFFIX = '.train_neg.csv' 
VALIDATION_NEG_SUFFIX = '.validation_neg.csv'  
TEST_NEG_SUFFIX = '.test_neg.csv'  
PROPENSITY_SUFFIX = '.propensity.npy'

VARIABLE_SUFFIX = '.variable.csv'  

DICT_SUFFIX = '.dict.csv'
DICT_POS_SUFFIX = '.dict_pos.csv'

C_HISTORY = 'history'  # history column name
C_HISTORY_LENGTH = 'history_length'  # history length column name
C_HISTORY_NEG = 'history_neg'  # negative history column name
C_HISTORY_NEG_LENGTH = 'history_neg_length'  # negative history length column name
C_HISTORY_POS_TAG = 'history_pos_tag'  # indicate positive 1 or negative 0
C_CTF_HISTORY = 'ctf_history' # counterfactual history column name
C_CTF_HISTORY_LENGTH = 'ctf_history_length'

# Text sequence
C_SENT = 'sent'  
C_WORD = 'word'  
C_WORD_ID = 'word_id'  
C_POS = 'pos'  
C_POS_ID = 'pos_id' 
C_TREE = 'tree' 
C_TREE_POS = 'tree_pos'

# # DataProcessor/feed_dict
X = 'x'
Y = 'y'
LABEL = 'label'
UID = 'uid'
IID = 'iid'
IIDS = 'iids'
TIME = 'time' 
RANK = 'rank'
REAL_BATCH_SIZE = 'real_batch_size'
TOTAL_BATCH_SIZE = 'total_batch_size'
TRAIN = 'train'
DROPOUT = 'dropout'
SAMPLE_ID = 'sample_id' 

# Hash
K_ANCHOR_USER = 'anchor_user' 
K_UID_SEG = 'uid_seg'  
K_SAMPLE_HASH_UID = 'sample_hash_pos'

# ProLogic
K_X_TAG = 'x_tag'  
K_OR_LENGTH = 'or_length'  
K_S_LENGTH = 'seq_length'  

# Syntax
K_T_LENGTH = 'tree_length'

# # out dict
PRE_VALUE = 'pre_value'
PREDICTION = 'prediction'  
CHECK = 'check' 
LOSS = 'loss' 
LOSS_L2 = 'loss_l2' 
EMBEDDING_L2 = 'embedding_l2'  
L2_BATCH = 'l2_batch'  
CTF_PREDICTION = 'ctf_prediction'
CTF_HIS_DIST = 'ctf_history_distance'
DIM = 'dim'
