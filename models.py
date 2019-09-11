def random(split):
    import numpy as np
    X_train, X_val, y_train, y_val = split
    np.random.seed(0)
    return np.random.rand(len(y_val))

def knn(split, nbors):
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_val, y_train, y_val = split
    clf = KNeighborsClassifier(n_neighbors=nbors)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:,1]

def logistic(split):
    from sklearn.linear_model import LogisticRegression
    X_train, X_val, y_train, y_val = split
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:,1]

def rf(split, n_estimators):
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_val, y_train, y_val = split
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:,1]
