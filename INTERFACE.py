# #!/usr/bin/env python
# # coding: utf-8

# # # Prediction Diabetes For Married Women Using Machine Learning Project

# # # Importing Important Libraries

# # In[1]:


# import numpy as np
# import pandas as pd 
# import statsmodels.api as sm
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import scale, StandardScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier


# In[32]:
import numpy as np
import pickle
model = pickle.load(open('trained_model.sav', 'rb'))
input_data = (0,84,82,31,125,38.2,0.233,23)
# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[33]:





# In[34]:


filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[35]:


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[ ]:




