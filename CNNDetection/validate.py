import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred_binary = y_pred > 0.5
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    # For real images (treating 0 as the positive class)
    r_precision = precision_score(y_true, y_pred_binary, pos_label=0)
    r_recall = recall_score(y_true, y_pred_binary, pos_label=0)
    r_f1 = f1_score(y_true, y_pred_binary, pos_label=0)

    # For fake images (treating 1 as the positive class)
    f_precision = precision_score(y_true, y_pred_binary, pos_label=1)
    f_recall = recall_score(y_true, y_pred_binary, pos_label=1)
    f_f1 = f1_score(y_true, y_pred_binary, pos_label=1)
    return acc, ap, r_acc, f_acc, r_precision, r_recall, r_f1, f_precision, f_recall, f_f1


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
