fs1_2_model = make_pipeline(
    Normalizer(norm="l1"),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.05, n_estimators=100), step=0.5),
    GaussianNB())

fs2_1_model = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=1.0, n_estimators=100), step=0.1),
    RBFSampler(gamma=0.35000000000000003),
    MLPClassifier(alpha=0.001, learning_rate_init=0.01, max_iter=600))
    
fs3_model = make_pipeline(
    MaxAbsScaler(),
    LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=0.001))

fs4_model = ExtraTreesClassifier(bootstrap=False, criterion="entropy", \
    max_features=0.6500000000000001, min_samples_leaf=20, min_samples_split=10, n_estimators=100)

hubert_model = make_pipeline(
    make_union(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        FunctionTransformer(copy)
    ),
    SelectFwe(score_func=f_classif, alpha=0.011),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.2, min_samples_leaf=2, min_samples_split=6, n_estimators=100))

hvm_fs2_model = make_pipeline(
    FeatureAgglomeration(affinity="euclidean", linkage="ward"),
    DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=9, min_samples_split=9))

regression_1_model = make_pipeline(
    make_union(
        make_pipeline(
            Normalizer(norm="l1"),
            StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=8, p=1, weights="uniform")),
            ZeroCount()
        ),
        FunctionTransformer(copy)
    ),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    StackingEstimator(estimator=RidgeCV()),
    Normalizer(norm="max"),
    RidgeCV()
)
