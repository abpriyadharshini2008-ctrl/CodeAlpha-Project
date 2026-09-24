

"""
CodeAlpha Data Analytics Internship
Task 2: Exploratory Data Analysis (EDA)
Task 3: Data Visualization
Dataset: diabetes.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# STEP 1: Load the dataset
# -------------------------------------------------
df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("STEP 1: FIRST LOOK AT THE DATA")
print("=" * 60)
print("\nFirst 5 rows:")
print(df.head())

print("\nShape of dataset (rows, columns):", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# -------------------------------------------------
# STEP 2: Data types and info
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA TYPES AND INFO")
print("=" * 60)
print(df.info())

# -------------------------------------------------
# STEP 3: Missing values check
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# -------------------------------------------------
# STEP 4: Statistical summary
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# -------------------------------------------------
# STEP 5: Outcome distribution (target column)
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: OUTCOME DISTRIBUTION")
print("=" * 60)
print(df['Outcome'].value_counts())
print("\nPercentage:")
print(df['Outcome'].value_counts(normalize=True) * 100)

# -------------------------------------------------
# STEP 6: Correlation between columns
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: CORRELATION WITH OUTCOME")
print("=" * 60)
correlation = df.corr()['Outcome'].sort_values(ascending=False)
print(correlation)

# -------------------------------------------------
# STEP 7: VISUALIZATIONS
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Diabetes Dataset - EDA & Visualization", fontsize=16, fontweight='bold')

# 1. Outcome count plot
sns.countplot(x='Outcome', data=df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Diabetes Outcome Count')
axes[0, 0].set_xticklabels(['No Diabetes', 'Diabetes'])

# 2. Age distribution
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Age Distribution')

# 3. Glucose vs Outcome (boxplot)
sns.boxplot(x='Outcome', y='Glucose', data=df, ax=axes[0, 2], palette='Set3')
axes[0, 2].set_title('Glucose Level vs Outcome')
axes[0, 2].set_xticklabels(['No Diabetes', 'Diabetes'])

# 4. Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')

# 5. BMI distribution by outcome
sns.histplot(data=df, x='BMI', hue='Outcome', kde=True, ax=axes[1, 1], palette='husl')
axes[1, 1].set_title('BMI Distribution by Outcome')

# 6. Scatter: Glucose vs BMI colored by Outcome
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, ax=axes[1, 2], palette='deep')
axes[1, 2].set_title('Glucose vs BMI')

plt.tight_layout()
plt.savefig('eda_results.png', dpi=150, bbox_inches='tight')
print("\n\nGraphs saved as 'eda_results.png' in the same folder!")

plt.show()

# -------------------------------------------------
# STEP 8: Key Insights (auto-printed summary)
# -------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
diabetic_pct = df['Outcome'].value_counts(normalize=True)[1] * 100
avg_glucose_diabetic = df[df['Outcome'] == 1]['Glucose'].mean()
avg_glucose_healthy = df[df['Outcome'] == 0]['Glucose'].mean()

print(f"1. {diabetic_pct:.1f}% of people in this dataset have diabetes.")
print(f"2. Average Glucose level (Diabetic): {avg_glucose_diabetic:.1f}")
print(f"3. Average Glucose level (Non-Diabetic): {avg_glucose_healthy:.1f}")
print(f"4. Strongest correlation with Outcome: {correlation.index[1]} ({correlation.iloc[1]:.2f})")
