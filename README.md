# EDA_EXP_4   Titanic Survival Analysis using Univariate Analysis

**Aim**

To perform univariate analysis on the Titanic dataset to understand the distribution and characteristics of individual variables (such as Age, Sex, Pclass, Fare, and Survived) and to draw insights about passengers and their survival patterns.


**Algorithm**

**1)Import Libraries:**

Load the required Python libraries (pandas, numpy, matplotlib, seaborn).

**2)Load the Dataset:**

Read the Titanic dataset from available sources (e.g., seaborn’s built-in Titanic dataset or a CSV file).

**3)Data Inspection:**

View the first few rows using head().

Get dataset summary using info() and describe().

**4)Handle Missing Data:**
Identify missing values using isnull().sum() and handle them appropriately (e.g., fill or drop).

**5)Univariate Analysis:**
Perform univariate analysis for each variable:

**Categorical Variables: (e.g., Sex, Pclass, Survived, Embarked)**
Use frequency tables and count plots.

**Numerical Variables: (e.g., Age, Fare)**
Use histograms, box plots, and summary statistics.

**6)Interpretation:**
Analyze distributions, central tendencies, and spread.
Identify patterns (e.g., more passengers in 3rd class, survival differences by gender).


**Program**

**Name : SUBIKSHA K

Reg No.: 212224040332**

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = sns.load_dataset('titanic')


print("🔹 First 5 Records:")
print(df.head())


print("\n🔹 Shape of dataset (rows, columns):", df.shape)


print("\n🔹 Missing values per column:")
print(df.isnull().sum())


print("\n🔹 Distribution of Sex:")
print(df['sex'].value_counts())
sns.countplot(x='sex', data=df)
plt.title("Distribution of Passengers by Sex")
plt.show()


survival_rate = df['survived'].mean() * 100
print(f"\n🔹 Percentage of Passengers Survived: {survival_rate:.2f}%")

sns.countplot(x='survived', data=df)
plt.title("Survival Distribution (0 = Not Survived, 1 = Survived)")
plt.show()


max_class = df['pclass'].value_counts().idxmax()
print(f"\n🔹 Passenger class with most passengers: Class {max_class}")
sns.countplot(x='pclass', data=df)
plt.title("Passenger Count by Class")
plt.show()


print("\n🔹 Passengers Embarked from Each Port:")
print(df['embarked'].value_counts())

sns.countplot(x='embarked', data=df)
plt.title("Passengers by Embarkation Port")
plt.show()


print("\n🔹 Deck with most passengers:")
print(df['deck'].value_counts())

sns.countplot(x='deck', data=df)
plt.title("Passenger Count by Deck")
plt.show()


plt.figure(figsize=(7,4))
sns.histplot(df['age'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

age_mean = df['age'].mean()
age_median = df['age'].median()
age_range = df['age'].max() - df['age'].min()

print(f"\n🔹 Age Mean: {age_mean:.2f}")
print(f"🔹 Age Median: {age_median:.2f}")
print(f"🔹 Age Range: {age_range:.2f}")


plt.figure(figsize=(6,3))
sns.boxplot(x=df['fare'])
plt.title("Boxplot of Passenger Fare")
plt.show()

print("\n Comment: The boxplot shows several outliers (very high fares), indicating a few very expensive tickets.")


plt.figure(figsize=(7,4))
sns.histplot(df['fare'], kde=True, bins=40, color='orange')
plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

fare_mean = df['fare'].mean()
fare_median = df['fare'].median()

print(f"\n🔹 Fare Mean: {fare_mean:.2f}")
print(f"🔹 Fare Median: {fare_median:.2f}")

if fare_mean > fare_median:
    print(" Inference: The distribution is RIGHT-SKEWED (long tail to the right).")
else:
    print(" Inference: The distribution is LEFT-SKEWED.")


print("\n INSIGHTS:")

print("""
1️ Most passengers were male, in 3rd class, and embarked from Southampton.
2️ Majority were aged between 20–40 years.
3️ Fares are highly skewed — a few passengers paid extremely high fares.
4️ Survival rate was around 38%.
5️ Women, children, and passengers from 1st class had higher survival chances.
6️ Univariate analysis helps us:
   - Detect data distribution patterns,
   - Identify outliers and missing data,
   - Understand categorical balance before deeper multivariate analysis.
""")



**Output**

<img width="766" height="372" alt="Screenshot 2025-11-12 093357" src="https://github.com/user-attachments/assets/23ffc597-9c42-4f3a-9a07-a3d6e4c26ba7" />

<img width="382" height="382" alt="Screenshot 2025-11-12 093435" src="https://github.com/user-attachments/assets/a7b6da9c-3a98-4de9-8d55-38e5bb799c7f" />

<img width="391" height="112" alt="Screenshot 2025-11-12 093512" src="https://github.com/user-attachments/assets/5b022638-892c-421d-b28f-855723252fc2" />

<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/11a770e7-42cb-4aa2-a3e3-fa341da34ce6" />

<img width="518" height="37" alt="Screenshot 2025-11-12 093615" src="https://github.com/user-attachments/assets/97318658-f4b1-496f-ad39-7f47fb3ff462" />

<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/0800d3d9-1994-4e01-95fd-0c0f198253c4" />

<img width="510" height="35" alt="Screenshot 2025-11-12 093658" src="https://github.com/user-attachments/assets/46fd5acc-c9b0-451e-8cf7-7d64193ee1ed" />

<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/440f06d3-ae45-4e6a-9f77-f745150fa6c4" />

<img width="418" height="129" alt="Screenshot 2025-11-12 093740" src="https://github.com/user-attachments/assets/f42d7fc6-b508-4664-988c-8c58d2052058" />

<img width="571" height="455" alt="download" src="https://github.com/user-attachments/assets/793bedd3-9523-49e4-b787-85f701adda6b" />

<img width="322" height="222" alt="Screenshot 2025-11-12 093954" src="https://github.com/user-attachments/assets/9b62818f-2d8b-42e5-a057-9384e52a8ec4" />

<img width="562" height="455" alt="download" src="https://github.com/user-attachments/assets/91632b30-d28c-4aa0-b1cf-076cebc94986" />

<img width="609" height="393" alt="download" src="https://github.com/user-attachments/assets/9b91c27f-8c44-4102-a531-45d695e2faed" />

<img width="246" height="75" alt="Screenshot 2025-11-12 094116" src="https://github.com/user-attachments/assets/87f2598b-1a3f-419d-af1a-6ecb51736941" />

<img width="489" height="316" alt="download" src="https://github.com/user-attachments/assets/5f59c87e-0709-48f6-8e63-d102cdabc125" />

<img width="618" height="393" alt="download" src="https://github.com/user-attachments/assets/95487cf5-981f-43ed-8bf3-3f616e3cc817" />

<img width="263" height="49" alt="Screenshot 2025-11-12 095159" src="https://github.com/user-attachments/assets/61197992-515d-4192-83a9-ad4db37b8782" />

<img width="775" height="291" alt="Screenshot 2025-11-12 095216" src="https://github.com/user-attachments/assets/1b4a4b9c-4077-433a-816d-ad64804d9644" />




**Result**
From the univariate analysis:

Majority of passengers were male and in 3rd class.

Around 38% survived, majority being females and higher-class passengers.

Age is right-skewed with most passengers aged 20–40 years.

Fare distribution shows a few high outliers for 1st class passengers.

Thus, univariate analysis helps understand the distribution and spread of each individual feature in the Titanic dataset before moving to bivariate or multivariate analysis.

Visualization:
Plot appropriate charts for each variable using Matplotlib/Seaborn.
