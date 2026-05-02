import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from sklearn.ensemble import RandomForestClassifier

# Optional SMOTE (safe)
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except:
    smote_available = False

print("🚀 Starting Fraud Detection Project...")

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("data/creditcard.csv")
print("Dataset Shape:", df.shape)

# ==============================
# 2. DISTRIBUTION
# ==============================
print("\nClass Distribution:\n", df["Class"].value_counts())

plt.figure()
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal")
plt.savefig("images/fraud_distribution.png")
plt.close()

# ==============================
# 3. SPLIT
# ==============================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 5. SMOTE (SAFE)
# ==============================
if smote_available:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", np.bincount(y_train))
else:
    print("Running without SMOTE")

# ==============================
# 6. MODEL (FAST)
# ==============================
model = RandomForestClassifier(
    n_estimators=50,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained")

# ==============================
# 7. PREDICTION
# ==============================
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# ==============================
# 8. EVALUATION
# ==============================
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 9. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.close()

# ==============================
# 10. ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("images/roc_curve.png")
plt.close()

# ==============================
# 11. PR CURVE
# ==============================
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("images/pr_curve.png")
plt.close()

# ==============================
# 12. SAVE MODEL
# ==============================
joblib.dump({
    "model": model,
    "scaler": scaler
}, "models/fraud_model.pkl")

print("✅ Model saved")

# ==============================
# 13. SAMPLE PREDICTIONS
# ==============================
print("\n🔍 Sample Predictions:")
for i in range(5):
    label = "🚨 FRAUD" if probs[i] > 0.8 else "✅ NORMAL"
    print(f"Transaction {i}: {label} ({probs[i]:.3f})")

print("\n🎉 DONE! Check 'images' + 'models'")