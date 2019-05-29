import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from stemming.porter2 import stem
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import itertools
def pre_process():
    print("Processing the data...")
    df = pd.read_csv('lyrics_final.csv')
    # df = df.groupby('genre', ).head(1000)
    df.replace('?', -9999999, inplace=True)
    dict_genres = {'Rock': 3, 'Country': 5, 'Hip-Hop': 1, 'Pop': 2, 'Jazz': 4}
    labels = []
    text = []
    genre = []
    for index, row in df.iterrows():
        labels.append(row[0])
        text.append(row[1])
    # stemming
    text = [[stem(word) for word in sentence.split(" ")] for sentence in text]
    text = [" ".join(sentence) for sentence in text]
    for index in labels:
        genre.append(dict_genres[index])
    print("Data processing done!!")
    print("Total songs: %s" % (len(labels)))
    return text, labels
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def train():
    f = open("multiclass.txt", "w")
    text, labels = pre_process()
    countVec = TfidfVectorizer(max_features=1500, stop_words='english', sublinear_tf=True, norm='l2')
    x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.3, random_state=0)
    x_trainCV = countVec.fit_transform(x_train)
    joblib.dump(countVec, "tfidf_vectorizer.pickle")
    x_testCV = countVec.transform(x_test)
    x_train = x_trainCV
    x_test = x_testCV

    #Decision Tree Classifier
    print("Decision Tree Classifier")
    f.write("Decision Tree Classifier \n")
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    accuracy=dt.score(x_test, y_test)
    print("accuracy for Decision Tree: %s" % (accuracy))
    f.write("accuracy for decision tree: %s \n" % (accuracy))

    #Random Forest Classifier
    print("Random Forest Classifier")
    # model = RandomForestClassifier(n_jobs=-1)
    # estimators = np.arange(900, 1000, 10)
    # scores = []
    # for n in estimators:
    #     model.set_params(n_estimators=n)
    #     model.fit(x_train, y_train)
    #     scores.append(model.score(x_test, y_test))
    # print(scores)
    f.write("Random Forest Classifier \n")
    rf = RandomForestClassifier(n_estimators=950, random_state=0).fit(x_train, y_train)
    joblib.dump(rf, "classifier.pickle")
    y_pred = rf.predict(x_test)
    accuracy = rf.score(x_test, y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("accuracy for Random Forest: %s" % (accuracy))
    f.write("accuracy for Random Forest: %s \n" % (accuracy))
    print('Non-normalized confusion matrix for Random Forest:')
    print(cnf_matrix)
    f.write("\nNon-normalized confusion matrix for Random Forest: \n")
    np.savetxt(f, cnf_matrix, fmt="%5d", delimiter="", newline="\n")


    normalized_confusion_matrix = cnf_matrix / cnf_matrix.sum(axis=1, keepdims=True)
    print('\n Normalized confusion matrix (with normalizatrion:)')
    print(normalized_confusion_matrix)
    f.write("\nNormalized confusion matrix for Random Forest: \n")
    np.savetxt(f, normalized_confusion_matrix, fmt="%6.2f", delimiter="", newline="\n")
    f.close()
    # Plot normalized confusion matrix
    class_names = ['Rock','Country','Hip-Hop','Pop','Jazz'];
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.show()


def test(input_string):
    vectorizer = joblib.load("tfidf_vectorizer.pickle")
    classifier = joblib.load("classifier.pickle")

    tr = vectorizer.transform(input_string)

    predictions = classifier.predict(tr)
    print(predictions[0])
    return predictions[0]
def test(input_string):
    vectorizer = joblib.load("tfidf_vectorizer.pickle")
    classifier = joblib.load("classifier.pickle")

    tr = vectorizer.transform(input_string)

    predictions = classifier.predict(tr)
    print(predictions[0])
    return predictions[0]

if __name__ == '__main__':
    train()
