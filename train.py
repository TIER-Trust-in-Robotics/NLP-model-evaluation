from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


"""EXAMPLE"""
from to_ds import IEMOCAP_LABELS, get_feature_names, load_iemocap_features

train_ds, test_ds, _, _ = load_iemocap_features(feature_set="all")
