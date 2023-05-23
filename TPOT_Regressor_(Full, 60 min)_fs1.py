import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -13.375462703772365
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, min_samples_split=2)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.55, tol=0.001)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=4, min_samples_split=7, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
