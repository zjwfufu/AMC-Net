from functools import reduce

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


def Visualize_ACM(model, signal, cfg, index):
    def hook_feature_map(module, input, output):
        feature_map.append(input[0].squeeze())
        feature_map.append(output.squeeze())

    feature_map = []

    model.ACM.register_forward_hook(hook=hook_feature_map)

    model.to('cpu')
    model(signal)

    fig1, axs1 = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axs1.reshape(-1)):
        ax.plot(feature_map[0][i, 0].detach().numpy())
        ax.plot(feature_map[0][i, 1].detach().numpy())

    fig1.savefig(cfg.result_dir + '/' + f'visualize_before_ACM#{index}.svg', format='svg', dpi=150)

    fig2, axs2 = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axs2.reshape(-1)):
        ax.plot(feature_map[1][i, 0].detach().numpy())
        ax.plot(feature_map[1][i, 1].detach().numpy())

    fig2.savefig(cfg.result_dir + '/' + f'visualize_after_ACM#{index}.svg', format='svg', dpi=150)


    plt.close()


def Draw_Confmat(Confmat_Set, snrs, cfg):
    for i, snr in enumerate(snrs):
        fig = plt.figure()
        df_cm = pd.DataFrame(Confmat_Set[i],
                             index=cfg.classes,
                             columns=cfg.classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        conf_mat_dir = os.path.join(cfg.result_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.svg', format='svg', dpi=150)
        plt.close()


def Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg):
    plt.plot(snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {cfg.dataset} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    acc_dir = os.path.join(cfg.result_dir, 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    plt.savefig(acc_dir + '/' + 'acc.svg', format='svg', dpi=150)
    plt.close()

    Accuracy_Mods = np.zeros((len(snrs), Confmat_Set.shape[-1]))

    for i, snr in enumerate(snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    for j in range(0, Confmat_Set.shape[-1]):
        plt.plot(snrs, Accuracy_Mods[:, j])

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {cfg.dataset} dataset")
    plt.grid()
    plt.legend(cfg.classes.keys())
    plt.savefig(acc_dir + '/' + 'acc_mods.svg', format='svg', dpi=150)
    plt.close()


def save_training_process(train_process, cfg):
    fig1 = plt.figure(1)
    plt.plot(train_process.epoch, train_process.lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate")
    plt.grid()
    fig1.savefig(cfg.result_dir + '/' + 'lr.svg', format='svg', dpi=150)
    plt.close()

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss,
             "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss,
             "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc,
             "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc,
             "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()
    fig2.savefig(cfg.result_dir + '/' + 'loss_acc.svg', format='svg', dpi=150)
    plt.show()
    plt.close()
