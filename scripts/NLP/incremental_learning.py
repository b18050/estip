# Run data cleaning process
# %run data_cleaning.ipynb

# Run model training
%run nlp_model.ipynb

# Pre incremental train test (model will predict negative)
input = ["I dont like this place, but it's good"]
print(input)
print(clf.predict(cv.transform(input)))

# Incremental training
X_instance = cv.transform(input)
y_instance = ['positive']
max_iter = 100

for i in range (0,max_iter):
    clf.partial_fit(X_instance, y_instance)
    if(clf.predict(X_instance) == y_instance):
        break


# Post incremental train test (weights changed so model will predict positive now)
print(input)
print(clf.predict(cv.transform(input)))

# Don't forget to save newly trained model
joblib.dump(cv, 'input_transformer.pkl')
joblib.dump(clf, 'review_sentiment.pkl')