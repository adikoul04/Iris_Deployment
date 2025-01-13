import joblib
clf = joblib.load("rf_model.sav")
print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))