import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def check_model(estimator, X, y, **model_params):
    results = cross_val_score(estimator, X, y, **model_params)
    print("Accuracy:")
    print("\tMedian:", np.median(results))
    print("\tMean:", np.mean(results))
    print("\tStd:", np.std(results))

def cross_val_score(estimator, X, y, size=0.1, **model_params):
    #assert X.shape[0] == y.shape[0]
    if not isinstance(y, np.ndarray):
        #X = np.array(X)
        y = np.array(y)
    size = int(X.shape[0]*size)
    results = []
    index = set(range(X.shape[0]))
    idx = np.arange(X.shape[0]).tolist()
    for i in tqdm(range(1, X.shape[0], size)):
        model = estimator(**model_params)
        tr_idx = idx[:i] + idx[i+size:]
        val_idx = idx[i:i+size]
        X_train = X[tr_idx]
        X_val = X[val_idx]
        y_train = y[tr_idx]
        y_val = y[val_idx]
        model.fit(X_train, y_train)
        results.append(accuracy_score(y_val.astype("int32"), model.predict(X_val)))
    return results