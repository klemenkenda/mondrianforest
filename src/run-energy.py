import numpy as np
import pprint as pp     # pretty printing module
from matplotlib import pyplot as plt        # required only for plotting results
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal 
from mondrianforest import process_command_line, MondrianForest

# ./mondrianforest_demo.py --dataset satimage --n_mondrians 100 --budget -1 
#    --normalize_features 1 --save 1 --data_path ../process_data/ --n_minibatches 10 
#    --store_every 1 --optype class

class settings:
    alpha = 0
    bagging = 0
    budget= -1.0
    budget_to_use= float("inf")
    data_path = 'D:\\MondrianForests\\data\\'
    dataset = 'id12041022_1_AR_WF_DT.arff'
    debug = 0
    discount_factor = 10
    draw_mondrian = 0
    init_id = 1
    min_samples_split = 2
    n_minibatches = 4000
    n_mondrians = 10
    name_metric = 'mse'
    normalize_features = 1
    op_dir = 'results'
    optype = 'real'
    perf_dataset_keys = ['train', 'test']
    perf_metrics_keys = ['log_prob', 'acc']
    perf_store_keys = ['pred_prob']
    save = 1
    select_features = 0
    smooth_hierarchically = 0
    store_every = 0
    tag = ''
    verbose = 1

def perform_file(fileName):
    settings.dataset = fileName + ".arff"
    print(fileName)
    
    print("Loading data")
    data = load_data(settings)
    
    param, cache = precompute_minimal(data, settings)
    mf = MondrianForest(settings, data)

    for idx_minibatch in range(settings.n_minibatches):    
        train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
            
        if idx_minibatch == 0:
            with open(settings.data_path + fileName + '.csv', 'w') as f:
                f.write("target;prediction\n")
            print("Training 0 batch", len(train_ids_current_minibatch))
            # Batch training for first minibatch
            mf.fit(data, train_ids_current_minibatch, settings, param, cache)
        else:        
            print('Evaluation on batch', idx_minibatch, 'in', fileName)
            # Evaluate
            weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians        
            results = mf.evaluate_predictions(data, data['x_train'][train_ids_current_minibatch], data['y_train'][train_ids_current_minibatch], settings, param, weights_prediction, True)
            # prediction
            predictions = results[0]['pred_mean']
            real = data['y_train'][train_ids_current_minibatch].flatten()
            for i in range(len(predictions)):
                # print(i, predictions[i], real[i])
                with open(settings.data_path + fileName + '.csv', 'a') as f:
                    f.write("{0},{1}\n".format(real[i], predictions[i]))        
            
            print("Training on next batch ", idx_minibatch, len(train_ids_current_minibatch))
            # Online update
            mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)
        print("Finished training ...")        

# main part of the programe
import optparse

parser = optparse.OptionParser();

parser.add_option('-n', '--node', action="store", dest="node", help="Name of the node to RF.", default="id12041022")
parser.add_option('-f', '--from', action="store", dest="fromH", help="From horizon", default=1)
parser.add_option('-t', '--to', action="store", dest="toH", help="To horizon", default=24)

options, args = parser.parse_args()

print("NODE:", options.node)
print("FROM:", options.fromH)
print("TO:", options.toH)

for horizon in range(int(options.fromH), int(options.toH) + 1):
    fileName = options.node + "_" + str(horizon) + "_AR_WF_DT"
    print(fileName)
    perform_file(fileName)