# Datheon Internship - Task 1
# Multi-Class Classification Pipeline: Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.base import clone

# 1. Load the dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
class_names = iris.target_names

df = X.copy()
df["species"] = y.map(dict(enumerate(class_names)))

print("=" * 55)
print("DATHEON INTERNSHIP: IRIS CLASSIFICATION")
print("=" * 55)
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df["species"].value_counts())

# 2. Exploratory Data Analysis
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Dataset - Feature Relationships", y=1.02)
plt.savefig("iris_pairplot.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Define two classification models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ))
    ])
}

# 5. Train, predict and evaluate
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    })

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)
    print(classification_report(
        y_test,
        predictions,
        target_names=class_names,
        zero_division=0
    ))

    # 6. Confusion matrix heatmap
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_confusion_matrix.png", dpi=300)
    plt.show()

# 7. Compare evaluation metrics
results_df = pd.DataFrame(results).set_index("Model")

print("\nMODEL COMPARISON")
print(results_df.round(4))

results_df.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

# 8. Decision boundary visualization using PCA
# This is a separate 2D visualization; the evaluation above
# uses all four original features.

X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit(X_train).transform(X_test)

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

print("\nVariance retained by two PCA components:",
      round(pca.explained_variance_ratio_.sum(), 4))

for name, original_model in models.items():
    # Fit a separate visualization model on the two PCA components.
    # This does not change the full-feature evaluation above.
    visual_model = clone(original_model)
    visual_model.fit(X_train_2d, y_train)

    x_min = X_train_2d[:, 0].min() - 1
    x_max = X_train_2d[:, 0].max() + 1
    y_min = X_train_2d[:, 1].min() - 1
    y_max = X_train_2d[:, 1].max() + 1

    import numpy as np

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = visual_model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.25, cmap="viridis")
    plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="viridis",
        edgecolors="black",
        s=45
    )
    plt.title(f"Decision Boundaries - {name} (PCA 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_decision_boundary.png", dpi=300)
    plt.show()

# 9. Save evaluation results
results_df.to_csv("classification_results.csv")

print("\nAll tasks completed successfully!")
print("Saved: classification_results.csv")
print("Saved: iris_pairplot.png")
print("Saved: model_comparison.png")
print("Confusion matrix and decision boundary plots saved.")