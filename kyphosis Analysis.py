#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df = pd.read_csv('housing.csv')


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.nunique()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df['total_bedrooms'].median()


# In[16]:


df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)


# In[17]:


for i in df.iloc[:,2:7]:
 df[i] = df[i].astype('int')
df.head()


# In[18]:


df.describe().T


# In[19]:


Numerical = df.select_dtypes(include=[np.number]).columns
print(Numerical)


# In[20]:


for col in Numerical:
 plt.figure(figsize=(10, 6))

 df[col].plot(kind='hist', title=col, bins=60, edgecolor='black')
 plt.ylabel('Frequency')

 plt.show()


# In[21]:


for col in Numerical:
 plt.figure(figsize=(6, 6))

 sns.boxplot(df[col], color='blue')
 plt.title(col)
 plt.ylabel(col)

 plt.show()


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['Target'] = data.target

# Display the first few rows
df.head()


# In[24]:


import pandas as pd

variable_meaning = {
    "MedInc": "Median income in block group",
    "HouseAge": "Median house age in block group",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Population of block group",
    "AveOccup": "Average number of household members",
    "Latitude": "Latitude of block group",
    "Longitude": "Longitude of block group",
    "Target": "Median house value (in $100,000s)"
}

# Create DataFrame from dictionary
variable_df = pd.DataFrame(list(variable_meaning.items()), columns=["Feature", "Description"])

print("\nVariable Meaning Table:")
print(variable_df)


# In[25]:


print("\nBasic Information about Dataset:")
print(df.info()) # Overview of dataset
print("\nFirst Five Rows of Dataset:")
print(df.head())


# In[26]:


print("\nSummary Statistics:")
print(df.describe())


# In[27]:


summary_explanation = """
The summary statistics table provides key percentiles and other descriptive metrics:

- **25% (First Quartile - Q1):** This represents the value below which 25% of the data falls. It's useful for identifying the lower range of the dataset.
- **50% (Median - Q2):** This is the middle value when the data is sorted. It provides a robust measure of central tendency that’s not affected by outliers.
- **75% (Third Quartile - Q3):** This represents the value below which 75% of the data falls. It helps understand the upper range of typical values.
- These percentiles are useful for detecting skewness, understanding data distribution, and identifying potential outliers.
"""

print("\nSummary Statistics Explanation:")
print(summary_explanation)


# In[28]:


print("\nMissing Values in Each Column:")
print(df.isnull().sum()) # Count of missing values


# In[29]:


plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()


# In[41]:


plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()


# In[42]:


plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[37]:


sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'Target']], diag_kind='kde')
plt.show()
print("1. The dataset has", df.shape[0], "rows and", df.shape[1], "columns.")
print("2. No missing values were found in the dataset.")
print("3. Histograms show skewed distributions in some features like 'MedInc'.")
print("4. Boxplots indicate potential outliers in 'AveRooms' and 'AveOccup'.")
print("5. Correlation heatmap shows 'MedInc' has the highest positive correlation with house value (Target).")


# In[47]:


import pandas as pd
data = pd.read_csv('Book1.csv')
print(data)


# In[48]:


def find_s_algorithm(data):
    """Implements the Find-S algorithm to find the most specific hypothesis."""

    # Extract feature columns and target column
    attributes = data.iloc[:, :-1].values  # All columns except the last
    target = data.iloc[:, -1].values       # Last column (class labels)

    # Step 1: Initialize hypothesis with first positive example
    for i in range(len(target)):
        if target[i].lower() == "yes":  # Case-insensitive match for "Yes"
            hypothesis = attributes[i].copy()
            break

    # Step 2: Update hypothesis based on other positive examples
    for i in range(len(target)):
        if target[i].lower() == "yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?'  # Generalize inconsistent attributes

    return hypothesis
final_hypothesis = find_s_algorithm(data)
# Print the learned hypothesis
print("Most Specific Hypothesis:", final_hypothesis)

# Example usage: assuming you have a DataFrame named `data`
# final_hypothesis = find_s_algorithm(data)
# print("Most Specific Hypothesis:", final_hypothesis)


# In[49]:


import pandas as pd
data = pd.read_csv('Book1.csv')
print(data)
def find_s_algorithm(data):
    """Implements the Find-S algorithm to find the most specific hypothesis."""

    # Extract feature columns and target column
    attributes = data.iloc[:, :-1].values  # All columns except the last
    target = data.iloc[:, -1].values       # Last column (class labels)

    # Step 1: Initialize hypothesis with first positive example
    for i in range(len(target)):
        if target[i].lower() == "yes":  # Case-insensitive match for "Yes"
            hypothesis = attributes[i].copy()
            break

    # Step 2: Update hypothesis based on other positive examples
    for i in range(len(target)):
        if target[i].lower() == "yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?'  # Generalize inconsistent attributes

    return hypothesis
final_hypothesis = find_s_algorithm(data)
# Print the learned hypothesis
print("Most Specific Hypothesis:", final_hypothesis)

# Example usage: assuming you have a DataFrame named `data`
# final_hypothesis = find_s_algorithm(data)
# print("Most Specific Hypothesis:", final_hypothesis)



# In[ ]:




