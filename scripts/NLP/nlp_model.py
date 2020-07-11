from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# Run data cleaning process
# %run data_cleaning.py

# Separate data and labels
X = df['review']
y = df['sentiment']

# Using a hashing vectorizer to keep model size low
cv = HashingVectorizer(stop_words='english', ngram_range=(1,2))
cv.fit(X)
X_fitted = cv.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_fitted, y, test_size=0.25, random_state=42)

# Linear SVM powered by SGD Classifier (params are defaults)
clf = SGDClassifier(loss="hinge", tol=None, max_iter=10)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cf_matrix, range(2),
                  range(2))

# plot (powered by seaborn)
ax= plt.subplot()
sn.set(font_scale=1)
sn.heatmap(df_cm, ax = ax, annot=True,annot_kws={"size": 16}, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix') 
ax.xaxis.set_ticklabels(['negative', 'positive'])
ax.yaxis.set_ticklabels(['negative', 'positive'])
plt.show()

# pickeling to save models
joblib.dump(cv, 'input_transformer.pkl')
joblib.dump(clf, 'review_sentiment.pkl')

