#from BM1 import BaseModelV1
from RegressionModel.BM_Avg_Decomp import BaseModelV1_Avg_deComp
from RegressionModel.BM1 import BaseModelV1
from RegressionModel.BM_Learning_Decomp import BaseModelV1_learning
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import *
import random

# from torch.utils.tensorboard import SummaryWriter


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

ts_size = 320

hidden_layers = [512, 512, 256, 128, 10]

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda used')
else:
    print('cpu used')
    device = 'cpu'


# train_file = "/Users/lanhai/PycharmProjects/CE4TS/Data/ecg_trainingData.npy"
# test_file = "/Users/lanhai/PycharmProjects/CE4TS/Data/ecg_testingData.npy"
# valid_file = "/Users/lanhai/PycharmProjects/CE4TS/Data/ecg_validationData.npy"

# train_file = "/research/local/hai/SyncCode/SelNetSync/data/ecg/train/ecg_trainingData.npy"
# test_file = "/research/local/hai/SyncCode/SelNetSync/data/ecg/train/ecg_testingData.npy"
# valid_file = "/research/local/hai/SyncCode/SelNetSync/data/ecg/train/ecg_validationData.npy"


train_file = "../data/ecg_trainingData.npy"
test_file = "../data/ecg_testingData.npy"
valid_file = "../data/ecg_validationData.npy"


def load_data(data_file):
    O = np.load(data_file)
    X = np.array(O[:, :ts_size], dtype=np.float32)
    T = []
    for rid in range(O.shape[0]):
        t = O[rid, ts_size]
        T.append([t])
    T = np.array(T)
    C = np.array(O[:, -1], dtype=np.float32)
    C.resize((X.shape[0], 1))
    return X, T, C


train_X, train_T, train_C = load_data(train_file)

test_X, test_T, test_C = load_data(test_file)

valid_X, valid_T, valid_C = load_data(valid_file)


def get_batch(batch_id, batch_size, X, T, C):
    train_num = X.shape[0]
    start_index = (batch_id * batch_size) % train_num
    end_index = start_index + batch_size

    batch_X = X[start_index: end_index]
    batch_T = T[start_index: end_index]
    batch_C = C[start_index: end_index]

    if batch_X.shape[0] < batch_size:
        L = batch_size - batch_X.shape[0]
        batch_X = np.concatenate((batch_X, X[:L]), axis=0)
        batch_T = np.concatenate((batch_T, T[:L]), axis=0)
        batch_C = np.concatenate((batch_C, C[:L]), axis=0)
    return torch.tensor(batch_X), torch.tensor(batch_T), torch.tensor(batch_C)


def mean_absolute_percentage_error(labels, predictions):
    return np.mean(np.abs((predictions - labels) * 1.0 / (labels + 0.000001))) * 100


def qerror_minmax(labels, predictions):
    #max_values, _ = torch.max(torch.concat((labels, predictions), dim=1), dim=1)
    #min_values, _ = torch.min(torch.concat((labels, predictions), dim=1), dim=1)
    max_values = np.maximum(labels, predictions)
    min_values = np.minimum(labels, predictions)
    q_error = max_values / min_values
    return np.mean(q_error)


def qerror(labels, predictions):
    return predictions / labels


def eval(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    q_error_minmax = qerror_minmax(labels, predictions)

    q_error = qerror(labels, predictions)
    underestimate_ratio = np.sum(q_error < 1) / len(q_error)
    overestimate_ratio = np.sum(q_error > 1) / len(q_error)
    average_overestimate = np.mean(q_error[q_error > 1]) - 1

    return (mse, mae, mape, q_error_minmax, underestimate_ratio, overestimate_ratio, average_overestimate)


def inference(ts_size, threshold_em_size, batch_size, model_path):
    BMV1 = BaseModelV1_learning(ts_size, threshold_em_size, hidden_layers)
    BMV1.load_state_dict(torch.load(model_path))
    BMV1.eval()
    n_batch_test = int(test_X.shape[0] / batch_size) + 1
    with torch.no_grad():
        predictions = None
        for b in range(n_batch_test):
            batch_X, batch_T, _ = get_batch(b, batch_size, test_X, test_T, test_C)
            batch_X.to(device=device)
            batch_T.to(device=device)
            # batch_C.to(device=device)

            #pred_batch = BMV1(batch_T, batch_X)

            pred_batch, pred_value = BMV1(batch_T, batch_X)
            pred_batch = pred_batch * 10 + pred_value

            pred_batch = pred_batch.cpu().numpy()
            if b == 0:
                predictions = pred_batch
            else:
                predictions = np.concatenate((predictions, pred_batch), axis=0)
        predictions = predictions[:test_X.shape[0]]
        predictions = np.hstack(predictions)
        test_C_labels = np.hstack(test_C)
        print("Loss {0}".format(eval(predictions, test_C_labels)))


def train(ts_size, threshold_em_size, learning_rate, batch_size, epochs, re_train, model_path, save=True):

    # writer = SummaryWriter('../train_runs/'+model_path)

    BMV1 = BaseModelV1_learning(ts_size, threshold_em_size, hidden_layers)

    print(BMV1)

    # initialization
    for m in BMV1.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.normal_(m.bias.data)

    if re_train == 1:
        BMV1.load_state_dict(torch.load(model_path))

    loss_fn = torch.nn.HuberLoss(delta=1.345)
    #loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(BMV1.parameters(), learning_rate, momentum=0.9)
    BMV1.to(device=device)

    current_best_score = float('inf')

    for epoch in range(epochs):
        print("epoch:{0}".format(epoch))
        BMV1.train()
        n_batch = int(train_X.shape[0] / batch_size) + 1
        for b in range(n_batch):
            batch_X, batch_T, batch_C = get_batch(b, batch_size, train_X, train_T, train_C)
            batch_X = batch_X.to(device=device)
            batch_T = batch_T.to(device=device)
            batch_C = batch_C.to(device=device)

            batch_C_range = torch.div(batch_C, 10)
            batch_C_value = batch_C % 10

            pred, pre_f2 = BMV1(batch_T, batch_X)

            range_loss = loss_fn(torch.log(pred), torch.log(batch_C_range))
            value_loss = loss_fn(torch.log(pre_f2), torch.log(batch_C_value))
            loss = range_loss + value_loss

            # pred = BMV1(batch_T, batch_X)
            #
            # loss = loss_fn(torch.log(pred), torch.log(batch_C))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        # test
        n_batch_valid = int(valid_X.shape[0] / batch_size) + 1
        BMV1.eval()
        with torch.no_grad():
            predictions = None
            for b in range(n_batch_valid):
                batch_X, batch_T, _ = get_batch(b, batch_size, valid_X, valid_T, valid_C)
                batch_X = batch_X.to(device=device)
                batch_T = batch_T.to(device=device)
                batch_C = batch_C.to(device=device)

                pred_batch, pred_value = BMV1(batch_T, batch_X)
                pred_batch = pred_batch * 10 + pred_value
                #pred_batch = BMV1(batch_T, batch_X)
                pred_batch = pred_batch.cpu().numpy()
                if b == 0:
                    predictions = pred_batch
                else:
                    predictions = np.concatenate((predictions, pred_batch), axis=0)
            predictions = predictions[:valid_X.shape[0]]
            predictions = np.hstack(predictions)
            valid_C_labels = np.hstack(valid_C)
            losses = eval(predictions, valid_C_labels)
            if losses[1] < current_best_score:
                torch.save(BMV1.state_dict(), model_path)
                current_best_score = losses[1]
                print("Current best model saved")

            print("Valid epoch {0} - Loss {1}".format(epoch, losses))

        # if save:
        #     torch.save(BMV1.state_dict(), model_path)
    # writer.close()


#train(ts_size, threshold_em_size=10, learning_rate=0.0002,
#      batch_size=216, epochs=500, re_train=0, model_path='../saved_model/ce4ts_base-learning-decomp-500-bs-216.model')
#
inference(ts_size, 10, 216, '../saved_model/ce4ts_base-learning-decomp-500-bs-216.model')
