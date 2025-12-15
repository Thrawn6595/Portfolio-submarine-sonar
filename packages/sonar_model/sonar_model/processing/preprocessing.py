from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import LabelEncoder

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder = None

def prepare_data_for_algorithm(X_train, X_test, y_train, y_test, encode_labels=False):
    Xtr = np.asarray(X_train)
    Xte = np.asarray(X_test)
    if not encode_labels:
        return PreparedData(
            X_train=Xtr, X_test=Xte,
            y_train=np.asarray(y_train).astype(int),
            y_test=np.asarray(y_test).astype(int),
            label_encoder=None
        )
    le = LabelEncoder()
    ytr = le.fit_transform(np.asarray(y_train))
    yte = le.transform(np.asarray(y_test))
    return PreparedData(X_train=Xtr, X_test=Xte, y_train=ytr.astype(int), y_test=yte.astype(int), label_encoder=le)
