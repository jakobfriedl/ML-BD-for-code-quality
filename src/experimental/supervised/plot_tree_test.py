from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from dtreeviz.trees import dtreeviz

wine = load_wine()
X = wine.data
y = wine.target

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=3,
                            max_features='auto',
                            min_samples_leaf=4,
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=0)
rf.fit(X, y)

viz = dtreeviz(rf.estimators_[99], X, y,
               target_name="wine category",
               feature_names=wine.feature_names,
               class_names=list(wine.target_names),
               title="100th decision tree - Wine data")

viz.save("decision_tree_wine.svg")