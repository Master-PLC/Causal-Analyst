import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, hamming_loss, label_ranking_loss, \
    precision_score, recall_score, roc_auc_score

from ...extras.misc import numpify

if TYPE_CHECKING:
    from transformers import EvalPrediction

warnings.filterwarnings("ignore")


@dataclass
class ComputeClassificationMetrics:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "accuracy": [], "item_accuracy": [], "precision": [], "recall": [], "f1": [], "hamming_loss": [], "hamming_score": [],
            "label_ranking_loss": [], "average_precision": [], "one_error": [], "auc": []
        }
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        logits = numpify(torch.sigmoid(eval_preds.predictions[0]))
        labels = numpify(eval_preds.predictions[1])
        preds = (logits > 0.5).astype(int)

        accuracy = accuracy_score(labels, preds)
        item_accuracy = item_accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='samples', zero_division=0)
        recall = recall_score(labels, preds, average='samples', zero_division=0)
        f1 = f1_score(labels, preds, average='samples', zero_division=0)
        hamming_loss_ = hamming_loss(labels, preds)
        hamming_score_ = 1 - hamming_loss_
        label_ranking_loss_ = label_ranking_loss(labels, logits)
        average_precision = average_precision_score(labels, logits, average='samples')
        one_error_ = one_error(labels, logits)
        auc = auc_score(labels, logits)

        self.score_dict["accuracy"].append(accuracy)
        self.score_dict["item_accuracy"].append(item_accuracy)
        self.score_dict["precision"].append(precision)
        self.score_dict["recall"].append(recall)
        self.score_dict["f1"].append(f1)
        self.score_dict["hamming_loss"].append(hamming_loss_)
        self.score_dict["hamming_score"].append(hamming_score_)
        self.score_dict["label_ranking_loss"].append(label_ranking_loss_)
        self.score_dict["average_precision"].append(average_precision)
        self.score_dict["one_error"].append(one_error_)
        self.score_dict["auc"].append(auc)

        if compute_result:
            return self._dump()


@dataclass
class ComputeGraphMetrics:

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "loss_nll": [], "loss_kl": [], "loss_elbo": [], "loss_sparse": [], "loss_lagr": [], "loss_mse": [], "loss_graph": [],
            "loss_align": [], "loss_cls": []
        }
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        losses = numpify(eval_preds.predictions)
        self.score_dict["loss_nll"].append(losses[:, 0])
        self.score_dict["loss_kl"].append(losses[:, 1])
        self.score_dict["loss_elbo"].append(losses[:, 2])
        self.score_dict["loss_sparse"].append(losses[:, 3])
        self.score_dict["loss_lagr"].append(losses[:, 4])
        self.score_dict["loss_mse"].append(losses[:, 5])
        self.score_dict["loss_graph"].append(losses[:, 6])
        self.score_dict["loss_align"].append(losses[:, 7])
        self.score_dict["loss_cls"].append(losses[:, 8])

        if compute_result:
            return self._dump()


@dataclass
class ComputeAllMetrics:

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "accuracy": [], "item_accuracy": [], "precision": [], "recall": [], "f1": [], "hamming_loss": [], "hamming_score": [],
            "label_ranking_loss": [], "average_precision": [], "one_error": [], "auc": [],
            "loss_nll": [], "loss_kl": [], "loss_elbo": [], "loss_sparse": [], "loss_lagr": [], "loss_mse": [], "loss_graph": [], 
            "loss_align": [], "loss_cls": [],
        }
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        logits = numpify(torch.sigmoid(eval_preds.predictions[0]))
        labels = numpify(eval_preds.predictions[1])
        preds = (logits > 0.5).astype(int)

        accuracy = accuracy_score(labels, preds)
        item_accuracy = item_accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='samples', zero_division=0)
        recall = recall_score(labels, preds, average='samples', zero_division=0)
        f1 = f1_score(labels, preds, average='samples', zero_division=0)
        hamming_loss_ = hamming_loss(labels, preds)
        hamming_score_ = 1 - hamming_loss_
        label_ranking_loss_ = label_ranking_loss(labels, logits)
        average_precision = average_precision_score(labels, logits, average='samples')
        one_error_ = one_error(labels, logits)
        auc = auc_score(labels, logits)

        self.score_dict["accuracy"].append(accuracy)
        self.score_dict["item_accuracy"].append(item_accuracy)
        self.score_dict["precision"].append(precision)
        self.score_dict["recall"].append(recall)
        self.score_dict["f1"].append(f1)
        self.score_dict["hamming_loss"].append(hamming_loss_)
        self.score_dict["hamming_score"].append(hamming_score_)
        self.score_dict["label_ranking_loss"].append(label_ranking_loss_)
        self.score_dict["average_precision"].append(average_precision)
        self.score_dict["one_error"].append(one_error_)
        self.score_dict["auc"].append(auc)

        losses = numpify(eval_preds.predictions[2])
        self.score_dict["loss_nll"].append(losses[:, 0])
        self.score_dict["loss_kl"].append(losses[:, 1])
        self.score_dict["loss_elbo"].append(losses[:, 2])
        self.score_dict["loss_sparse"].append(losses[:, 3])
        self.score_dict["loss_lagr"].append(losses[:, 4])
        self.score_dict["loss_mse"].append(losses[:, 5])
        self.score_dict["loss_graph"].append(losses[:, 6])
        self.score_dict["loss_align"].append(losses[:, 7])
        self.score_dict["loss_cls"].append(losses[:, 8])

        if compute_result:
            return self._dump()


def item_accuracy_score(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
        # print(f'Accuracy: {p / q:.4f}')
    return count / y_true.shape[0]


def auc_score(y_true, y_score):
    n_samples = y_true.shape[0]
    aucs = []

    for i in range(n_samples):
        auc = roc_auc_score(y_true[i], y_score[i])
        aucs.append(auc)

    return np.mean(aucs)


def one_error(y_true, y_score):
    n_samples = y_true.shape[0]
    errors = 0

    for i in range(n_samples):
        pred_label = np.argmax(y_score[i])  # 找到最高分数标签
        if y_true[i, pred_label] != 1:      # 检查是否在真实标签集合中
            errors += 1

    return errors / n_samples