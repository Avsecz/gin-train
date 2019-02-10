import sklearn.metrics as skm
import pandas as pd
import numpy as np
from collections import OrderedDict
import gin

# --------------------------------------------
# Combined metrics


@gin.configurable
class BootstrapMetric:
    def __init__(self, metric, n):
        """
        Args:
          metric: a function accepting (y_true and y_pred) and
             returning the evaluation result
          n: number of bootstrap samples to draw
        """
        self.metric = metric
        self.n = n

    def __call__(self, y_true, y_pred):
        outl = []
        for i in range(self.n):
            bsamples = (
                pd.Series(np.arange(len(y_true))).sample(frac=1, replace=True).values
            )
            outl.append(self.metric(y_true[bsamples], y_pred[bsamples]))
        return outl


@gin.configurable
class MetricsList:
    """Wraps a list of metrics into a single metric returning a list"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [metric(y_true, y_pred) for metric in self.metrics]


@gin.configurable
class MetricsDict:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return {k: metric(y_true, y_pred) for k, metric in self.metrics.items()}


@gin.configurable
class MetricsTupleList:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [(k, metric(y_true, y_pred)) for k, metric in self.metrics]


@gin.configurable
class MetricsOrderedDict:
    """Wraps a OrderedDict/tuple list of metrics into a single metric
    returning an OrderedDict
    """

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return OrderedDict([(k, metric(y_true, y_pred)) for k, metric in self.metrics])


@gin.configurable
class MetricsMultiTask:
    """Run the same metric across multiple tasks
    """

    def __init__(self, metrics, task_names=None):
        self.metrics = metrics
        self.task_names = task_names

    def __call__(self, y_true, y_pred):
        n_tasks = y_true.shape[1]
        if self.task_names is None:
            self.task_names = [i for i in range(n_tasks)]
        else:
            assert len(self.task_names) == n_tasks
        return OrderedDict([(task, self.metrics(y_true[:, i], y_pred[:, i]))
                            for i, task in enumerate(self.task_names)])


@gin.configurable
class MetricsAggregated:

    def __init__(self,
                 metrics,
                 agg_fn={"mean": np.mean, "std": np.std},
                 prefix=""):
        self.metrics
        self.agg_fn = agg_fn
        self.prefix = prefix

    def __call__(self, y_true, y_pred):
        out = self.metrics(y_true, y_pred)
        # TODO - generalize using numpy_collate?
        m = np.array(list(out.values()))
        return {self.prefix + k: fn(m) for k, fn in self.agg_fn}


@gin.configurable
class MetricsConcise:

    def __init__(self, metrics):
        import concise
        self.metrics_dict = OrderedDict([(m, concise.eval_metrics.get(m))
                                         for m in metrics])

    def __call__(self, y_true, y_pred):
        return OrderedDict([(m, fn(y_true, y_pred))
                            for m, fn in self.metrics_dict.items()])


# -----------------------------
# Binary classification
# Metric helpers
MASK_VALUE = -1
# Binary classification


def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)


@gin.configurable
def n_positive(y_true, y_pred):
    return y_true.sum()


@gin.configurable
def n_negative(y_true, y_pred):
    return (1 - y_true).sum()


@gin.configurable
def frac_positive(y_true, y_pred):
    return y_true.mean()


@gin.configurable
def accuracy(y_true, y_pred, round=True):
    """Classification accuracy
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.accuracy_score(y_true, y_pred)


@gin.configurable
def auc(y_true, y_pred, round=True):
    """Area under the ROC curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)

    if round:
        y_true = y_true.round()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return skm.roc_auc_score(y_true, y_pred)


@gin.configurable
def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)


@gin.configurable
def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)


@gin.configurable
def f1(y_true, y_pred, round=True):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.f1_score(y_true, y_pred)


@gin.configurable
def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))


classification_metrics = [
    ("auPR", auprc),
    ("auROC", auc),
    ("accuracy", accuracy),
    ("n_positive", n_positive),
    ("n_negative", n_negative),
    ("frac_positive", frac_positive),
]


@gin.configurable
class ClassificationMetrics:
    """All classification metrics
    """
    cls_metrics = classification_metrics

    def __init__(self):
        self.classification_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        return self.classification_metric(y_true, y_pred)
# TODO - add gin macro for a standard set of classification and regession metrics


# --------------------------------------------
# Regression

@gin.configurable
def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]


@gin.configurable
def kendall(y_true, y_pred, nb_sample=100000):
    """Kendall's tau coefficient, Kendall rank correlation coefficient
    """
    from scipy.stats import kendalltau
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) > nb_sample:
        idx = np.arange(len(y_true))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    return kendalltau(y_true, y_pred)[0]


@gin.configurable
def mad(y_true, y_pred):
    """Median absolute deviation
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


@gin.configurable
def rmse(y_true, y_pred):
    """Root mean-squared error
    """
    return np.sqrt(mse(y_true, y_pred))


@gin.configurable
def rrmse(y_true, y_pred):
    """1 - rmse
    """
    return 1 - rmse(y_true, y_pred)


@gin.configurable
def mse(y_true, y_pred):
    """Mean squared error
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean(axis=None)


@gin.configurable
def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


@gin.configurable
def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


@gin.configurable
def pearsonr(y_true, y_pred):
    from scipy.stats import pearsonr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return pearsonr(y_true, y_pred)[0]


@gin.configurable
def spearmanr(y_true, y_pred):
    from scipy.stats import spearmanr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return spearmanr(y_true, y_pred)[0]


regression_metrics = [
    ("mse", mse),
    ("var_explained", var_explained),
    ("pearsonr", pearsonr),  # pearson and spearman correlation
    ("spearmanr", spearmanr),
    ("mad", mad),  # median absolute deviation
]


@gin.configurable
class RegressionMetrics:
    """All classification metrics
    """
    cls_metrics = regression_metrics

    def __init__(self):
        self.regression_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        # squeeze the last dimension
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = np.ravel(y_true)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        return self.regression_metric(y_true, y_pred)


# available eval metrics --------------------------------------------


BINARY_CLASS = ["auc", "auprc", "accuracy", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION
