import torch
import warnings
from colorama import init, Fore, Back, Style
import argparse
import random
import numpy as np
from preprocess import load_data
from statistics import mean, stdev
from model import MyGHU
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm
from colorama import init, Fore, Back, Style
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def my_train(data_feature, data_feature_4000, data_label,
             data_graph, data_graph_4000, train_index, test_index, args, fold_num):
    acc_MLP_current = 0
    acc_GCN_current = 0

    nb_dim = data_feature.size(-1)
    nb_dim_4000 = data_feature_4000.size(-1)
    # data_feature_4000 = F.normalize(data_feature_4000, p=1, dim=0)
    nb_classes = (data_label.max()-data_label.min() + 1).item()

    model_GHU = MyGHU(nb_dim_4000, nb_dim, [128, 64, 16], [64, 32, 16], view=5, cls_number=nb_classes).to(device)
    optimiser_GHU = torch.optim.Adam(model_GHU.parameters(), lr=args.lr)

    celoss = torch.nn.CrossEntropyLoss()
    data_label_train = data_label[train_index]
    data_label_test = data_label[test_index]
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for current_iter, epoch in enumerate(tqdm(range(0, args.epochs + 1))):
        model_GHU.train()
        optimiser_GHU.zero_grad()
        P = model_GHU(data_feature_4000, data_graph_4000, data_feature, data_graph)
        loss = celoss(P[train_index], data_label_train)

        loss.backward()
        optimiser_GHU.step()

        #################################
        if True: #epoch % 20 == 0 and epoch != 0:
            model_GHU.eval()

            P = model_GHU(data_feature_4000, data_graph_4000, data_feature, data_graph)

            P = P[test_index].data.max(1)[1]
            P_T = data_label_test.cpu()

            accs = accuracy_score(P_T, P.cpu())
            if nb_classes != 2:
                precision = accs
                recall = accs
                f1 = accs
                auc = accs
            else:
                precision = precision_score(P_T, P.cpu())
                recall = recall_score(P_T, P.cpu())
                f1 = f1_score(P_T, P.cpu())
                try:
                    auc = roc_auc_score(P_T, P.cpu())
                except:
                    auc = f1 * 0
            string_2 = Fore.GREEN + " epoch: {} MLP,acc: {:.3f},pre: {:.3f},recall: {:.3f},f1: {:.3f},auc: {:.3f}".format(
                epoch, accs, precision, recall, f1, auc)

            acc_list.append(accs)
            auc_list.append(auc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            # writer_tb.add_scalar('Global_ACC', accs, epoch)
            # writer_tb.add_scalar('Global_Auc', auc, epoch)
            # writer_tb.add_scalar('Global_Pre', precision, epoch)
            # writer_tb.add_scalar('Global_Recall', recall, epoch)
            # writer_tb.add_scalar('Global_F1', f1, epoch)

    return acc_list, auc_list , precision_list, recall_list, f1_list




def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # load_Gan_data3(dataset='FTD_fourF')
    full_data = load_data(args.dataset, args.seed, args.fold, remake = False, splite= args.split)

    N = full_data['n']
    data_feature = torch.from_numpy(full_data['ori_graph']).cuda(device) #torch.from_numpy(full_data['data_feature']).cuda(device)
    data_feature_4000 = torch.from_numpy(full_data['data_feature_4000']).cuda(device)
    data_label= torch.from_numpy(full_data['data_label']).long().cuda(device).squeeze()
    # ori_graph = torch.from_numpy(full_data['ori_graph']).cuda(device)
    data_graph = torch.from_numpy(full_data['data_graph']).cuda(device)
    data_graph_4000 = torch.from_numpy(full_data['data_graph_4000']).cuda(device)
    train_list = full_data['train_list']
    test_list = full_data['test_list']
    fold_num = 0

    acc_list_mlp = []
    auc_list_mlp = []
    precision_list_mlp = []
    recall_list_mlp = []
    f1_list_mlp = []


    lable_matrix = (data_label.view(N, 1).repeat(1, N) == data_label.view(1, N).repeat(
        N, 1)) + 0
    data_graph_4000_see =  data_graph_4000 - torch.diag_embed(data_graph_4000.diag())
    acc_matrix = (lable_matrix * data_graph_4000_see).sum().item() / data_graph_4000_see.sum().item()
    print("acc_matrix: {:.2f}".format(acc_matrix * 100))

    for train_index, test_index in zip(train_list, test_list):
        fold_num+=1
        train_index = torch.from_numpy(train_index).long().numpy().tolist()
        test_index = torch.from_numpy(test_index).long().numpy().tolist()
        acc_list, auc_list , precision_list, recall_list, f1_list  = my_train(data_feature, data_feature_4000, data_label,
                              data_graph, data_graph_4000, train_index, test_index, args, fold_num)

        # if write_flag:
        #     writer.write({'epoch': 1000, "fold": fold_num},
        #                  {"test_acc": np.array(acc_list[-2:]).mean(),
        #                   "test_auc": np.array(auc_list[-2:]).mean(),
        #                   "test_precision": np.array(precision_list[-2:]).mean(),
        #                   "test_recall": np.array(recall_list[-2:]).mean(),
        #                   "test_f1": np.array(f1_list[-2:]).mean(),
        #                   })
        # # acc_list_gcn.append(global_acc_gcn)
        # acc_list_mlp.append(np.array(acc_list[-2:]).mean())
        # auc_list_mlp.append(np.array(auc_list[-2:]).mean())
        # precision_list_mlp.append(np.array(precision_list[-2:]).mean())
        # recall_list_mlp.append(np.array(recall_list[-2:]).mean())
        # f1_list_mlp.append(np.array(f1_list[-2:]).mean())
        # print("="*10 +str(fold_num)+"_End"+"="*10)

    # if write_flag:
    #     writer.write({'epoch': 0, "fold": -1},
    #                  {"test_acc": np.array(acc_list_mlp).mean(),
    #                   "test_auc": np.array(auc_list_mlp).mean(),
    #                   "test_precision": np.array(precision_list_mlp).mean(),
    #                   "test_recall": np.array(recall_list_mlp).mean(),
    #                   "test_f1": np.array(f1_list_mlp).mean(),
    #                   })
    # print(string_end2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specify for dataset site
    parser.add_argument('--split', type=int, default=20, help='label rate')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--res_dir', type=str, default='./result/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='ADNI_2C_40S')#, choices=['FTD_40FS','ADNI_3C_40S','ADNI_4C_40S','ADNI_2C_40S','ADNI','FTD_fourF', 'FTD'])  # abide
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--fold', type=float, default=5)

    args = parser.parse_args()


    # import socket, getpass, os
    # host_name = socket.gethostname()
    # from sql_writer import WriteToDatabase, get_primary_key_and_value, get_columns
    # TABLE_NAME = '111'
    # PRIMARY_KEY, PRIMARY_VALUE = get_primary_key_and_value(
    #     {
    #         'split': ['integer', args.split],
    #         "seed": ["integer", args.seed],
    #         "lr": ["double precision", args.lr],
    #         "dataset": ["text", args.dataset],
    #         "epoch": ["integer", None],
    #         "fold": ["integer", None],
    #         "model_name": ["text", host_name+os.path.split(__file__)[-1][:-3]]
    #     }
    # )
    # REFRESH = False
    # OVERWRITE = True
    #
    # test_val_metrics = {
    #     "acc": None,
    #     "auc": None,
    #     "precision": None,
    #     "recall": None,
    #     "f1": None,
    # }
    # train_val_metrics = {
    #     "pass": None,
    # }


    main(args)
