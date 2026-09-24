import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

print("Loading dataset...")
df = pd.read_csv("diabetes.csv")
print(f"Shape: {df.shape}")
print(df.head())

# Replace 0s with NaN for columns where 0 is not valid
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Fill missing values with median
for col in df.columns:
    df[col] = df[col].fillna(df[col].median())

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
print("All clean!")

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']
print(f"X shape: {X.shape}")
print(f"Target (0=No Diabetes, 1=Diabetes):\n{y.value_counts()}")

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'acc': acc,
        'roc': roc
    }
    print(f"Accuracy: {acc*100:.2f}%  ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Disease Prediction Model Results', fontsize=15, fontweight='bold')
colors = ['#e74c3c', '#2ecc71', '#3498db']

# Confusion matrices
for idx, (name, res) in enumerate(results.items()):
    ax = axes[0][idx]
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# ROC Curves
ax_roc = axes[1][0]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={res['roc']:.3f})", color=color, lw=2)
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title('ROC Curves', fontweight='bold')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(fontsize=9)
ax_roc.grid(alpha=0.3)

# Accuracy Comparison
ax_acc = axes[1][1]
names = list(results.keys())
accs = [res['acc'] * 100 for res in results.values()]
bars = ax_acc.bar(names, accs, color=colors, edgecolor='black')
ax_acc.set_title('Accuracy Comparison', fontweight='bold')
ax_acc.set_ylabel('Accuracy (%)')
ax_acc.set_ylim([0, 115])
ax_acc.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
for bar, acc in zip(bars, accs):
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontweight='bold')

# Feature Importances
ax_fi = axes[1][2]
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=X.columns)
fi.nlargest(8).sort_values().plot(kind='barh', ax=ax_fi, color='#e67e22', edgecolor='black')
ax_fi.set_title('Feature Importances (Random Forest)', fontweight='bold')
ax_fi.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('disease_results.png', dpi=150, bbox_inches='tight')
plt.show()

best = max(results, key=lambda k: results[k]['roc'])
print(f"\nBest Model : {best}")
print(f"Accuracy   : {results[best]['acc']*100:.2f}%")
print(f"ROC-AUC    : {results[best]['roc']:.4f}")
print("Done! Chart saved as disease_results.png")
