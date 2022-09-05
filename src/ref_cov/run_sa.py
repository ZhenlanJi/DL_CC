import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def infog(msg):
    return Colors.OKGREEN + msg + Colors.ENDC


def info(msg):
    return Colors.OKBLUE + msg + Colors.ENDC


def warn(msg):
    return Colors.WARNING + msg + Colors.ENDC


def fail(msg):
    return Colors.FAIL + msg + Colors.ENDC


def compute_roc(probs_neg, probs_pos):
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def compute_roc_auc(test_sa, adv_sa, split=1000):
    tr_test_sa = np.array(test_sa[:split])
    tr_adv_sa = np.array(adv_sa[:split])

    tr_values = np.concatenate(
        (tr_test_sa.reshape(-1, 1), tr_adv_sa.reshape(-1, 1)), axis=0
    )
    tr_labels = np.concatenate(
        (np.zeros_like(tr_test_sa), np.ones_like(tr_adv_sa)), axis=0
    )

    lr = LogisticRegressionCV(cv=5, n_jobs=-1).fit(tr_values, tr_labels)

    ts_test_sa = np.array(test_sa[split:])
    ts_adv_sa = np.array(adv_sa[split:])
    values = np.concatenate(
        (ts_test_sa.reshape(-1, 1), ts_adv_sa.reshape(-1, 1)), axis=0
    )
    labels = np.concatenate(
        (np.zeros_like(ts_test_sa), np.ones_like(ts_adv_sa)), axis=0
    )

    probs = lr.predict_proba(values)[:, 1]

    _, _, auc_score = compute_roc(
        probs_neg=probs[: (len(test_sa) - split)],
        probs_pos=probs[(len(test_sa) - split) :],
    )

    return auc_score

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.
        model = load_model("./model/model_mnist.h5")
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["activation_3"]

        # Load target set.
        x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model("./model/model_cifar.h5")
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    if args.lsa:
        test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)

        target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
        )

        auc = compute_roc_auc(test_lsa, target_lsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    if args.dsa:
        test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)

        target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        )

        auc = compute_roc_auc(test_dsa, target_dsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    print(infog("{} coverage: ".format(args.target) + str(target_cov)))
