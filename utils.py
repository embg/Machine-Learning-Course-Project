import matplotlib.pyplot as plt
import numpy as np

# Compute the (interpolated) ROC of a list of probabilities y_probs
def compute_roc(split, y_probs):
    from sklearn.metrics import roc_curve
    X_train, X_val, y_train, y_val = split

    fpr_axis = np.linspace(0, 1)
    fpr, tpr, _ = roc_curve(y_val, y_probs, pos_label = 1)
    curve = np.interp(fpr_axis, fpr, tpr)
    curve[0] = 0.0
    return curve

# Plot a mean ROC curve and individual curves for the arrays fprs, tprs
def plot_roc_curves(curves, color='b', mean_only=False, show=True, label=''):
    mean_curve = np.array(curves).mean(axis=0)
    plt.plot(np.linspace(0,1), mean_curve, color,label=label)
    if mean_only == False:
        for curve in curves:
            plt.plot(np.linspace(0,1), curve, color, alpha=0.25)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if show:
        plt.show()

# Run CV splits on model
def cross_validate(X, y, model, params=None, num_splits=5):
    from sklearn.model_selection import KFold
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=0).split(X)
    curves = []
    np.random.seed(0)
    
    for i, (train, val) in enumerate(splits):
    #    print(f'Split {i} of {num_splits}.')
        split = X.iloc[train], X.iloc[val], y.iloc[train], y.iloc[val]
        #  y_probs = models.knn_model(split, 400)
        if params is not None:
            y_probs = model(split, *params)
        else:
            y_probs = model(split)
        curve = compute_roc(split, y_probs)
        curves.append(curve)

    return curves

def confusion_matrix(X, y, model, params=None, num_splits=5):
    from sklearn.model_selection import KFold
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=0).split(X)
    np.random.seed(0)
    
    for i, (train, val) in enumerate(splits):
        split = X.iloc[train], X.iloc[val], y.iloc[train], y.iloc[val]
        if params is not None:
            y_probs = model(split, *params)
        else:
            y_probs = model(split)
        y_pred = np.rint(y_probs) # round probabilities to 0/1
        break
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y.iloc[val], y_pred)

def auc(curves):
    mean_curve = np.array(curves).mean(axis=0)
    return sum(mean_curve) / len(mean_curve)
