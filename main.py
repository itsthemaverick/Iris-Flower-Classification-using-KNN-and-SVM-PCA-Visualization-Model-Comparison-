from src.load_data import load_data
from src.preprocess import preprocess
from src.train_knn import train_knn
from src.train_svm import train_svm
from src.evaluate import evaluate
from src.visualize import plot_decision_boundary

X, y = load_data("data/iris.csv")

X_train, X_test, y_train, y_test = preprocess(X, y)

knn_model = train_knn(X_train, y_train, k=5)
svm_model = train_svm(X_train, y_train)

knn_metrics = evaluate(knn_model, X_test, y_test)
svm_metrics = evaluate(svm_model, X_test, y_test)

print("KNN Accuracy:", knn_metrics["accuracy"])
print(knn_metrics["report"])

print("SVM Accuracy:", svm_metrics["accuracy"])
print(svm_metrics["report"])

plot_decision_boundary(knn_model, X_train, y_train, "KNN Decision Boundary (PCA)")
plot_decision_boundary(svm_model, X_train, y_train, "SVM Decision Boundary (PCA)")
