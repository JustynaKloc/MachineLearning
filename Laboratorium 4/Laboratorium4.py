#!/usr/bin/env python
# coding: utf-8

# Ćwiczenie:
# 
# - pobierz zbiór danych Rain in Australia (załączony do tej instrukcji)
# 
# - usuń kolumny mające więcej niż 30% brakujących wartości oraz kolumnę 'Risk-MM'
# 
# - dokonaj imputacji brakujących wartości zakładając że są one MCAR (Missing Completely At Random), tzn. zmienne kategoryczne należy zastąpić dominantą (najczęściej występującą w kolumnie wartością) a numeryczne medianą (wartością środkową w rosnąco posortowanej kolumnie)
# 
# - obetnij obserwacje odstające więcej niż 1.5 rozstępu ćwiartkowego
# 
# - znormalizuj (numeryczne) i zakoduj (kategoryczne) dane
# 
# - wykorzystując sklearn.linear_model.LogisticRegression, dla każdego z regionów naucz model przewidujący jutrzejsza pogodę 'RainTomorrow'
# 
# - sprawdź który z modeli najskuteczniej przewiduje pogodę w skali całego kraju (dla każdego klasyfikatora zbiór testowy powinien być próbkowany z podzbiorów testowych dla wszystkich regionów)
# 
# - sprawdź czy wybrany model miał najwyższe accuracy na własnym zbiorze testowym spośród wszystkich klasyfikatorów
# 
# - porównaj najlepszy klasyfikator z "klasyfikatorem" zawsze wybierającym dominującą wartość w zbiorze
# 
# - dla najlepszego klasyfikatora wyświetl confusion matrix
# 
# Przy wykonywaniu zadania należy pamiętać o stratyfikacji.

# Import bibliotek

# In[1]:


import pandas as pd
import numpy as np


# Wczytanie bazy danych

# In[2]:


rain = pd.read_csv('weatherAUS.csv')


# Usunięcie kolumny Risk-MM

# In[3]:


rain =rain.drop(['RISK_MM'],axis=1)


# Usunięcie kolumn, w których brakuje więcej niż 30% wartości

# In[4]:


raindf = rain.drop(rain.loc[:,list((100*(rain.isnull().sum()/len(rain.index))>30))].columns, 1)
raindf.head()


# Sprawdzenie, które kolumny zawierają nullowe wartości:

# In[5]:


columns = []
for col in raindf.columns: 
    columns.append(col) 


# In[6]:


for columna in columns: 
    print(raindf[columna].isnull().sum())


# In[7]:


kolumny = columns[2:-1]
kolumny


# In[8]:


dominanty= []
for kolumna in kolumny:
    a = raindf[kolumna].value_counts().nlargest(1)
    dominanty.append(a)


# In[9]:


dominanty


# In[10]:


raindf.head()


# Uzupełnienie zmiennych kategorycznych dominantami

# In[11]:


raindf['RainToday'].fillna('No', inplace = True)
raindf['WindDir3pm'].fillna('SE', inplace = True)
raindf['WindDir9am'].fillna('N', inplace = True)
raindf['WindGustDir'].fillna('W', inplace = True)


# Uzupełnienie zmiennych numerycznych medianą 

# In[12]:


raindf.fillna(raindf.median(),inplace = True)


# Sprawdzenie czy zostały jeszcze wartości Nan

# In[13]:


columns = []
for col in raindf.columns: 
    columns.append(col) 
for columna in columns: 
    print(raindf[columna].isnull().sum())
kolumny_puste = [i for i in columns ]


# obcięcie obserwacji odstających więcej niż 1.5 rozstępu ćwiartkowego

# In[14]:


newdf = raindf.select_dtypes(include='float64')
columns_n = []
for col in newdf.columns: 
    columns_n.append(col) 


# In[15]:


columns_n


# In[21]:


for kolumna in columns_n:
    Q1 = raindf[kolumna].quantile(0.25)
    Q3 = raindf[kolumna].quantile(0.75)
    IQR = Q3 - Q1
    A = Q1 - 1.5 * IQR 
    B = Q3 + 1.5 * IQR
    mask = raindf[kolumna].between(A, B, inclusive=True)
    for i in range(len(mask)):
        if (mask[i] == False):
            raindf = raindf.drop([i], axis=0)     


# In[22]:


raindf.shape


# In[23]:


len(mask)


# znormalizuj (numeryczne) i zakoduj (kategoryczne) dane

# wykorzystując sklearn.linear_model.LogisticRegression, dla każdego z regionów naucz model przewidujący jutrzejsza pogodę 'RainTomorrow'

# In[34]:


Q1 = raindf['MinTemp'].quantile(0.25)
Q3 = raindf['MinTemp'].quantile(0.75)
IQR = Q3 - Q1
A = Q1 - 1.5 * IQR 
B = Q3 + 1.5 * IQR
mask = raindf['MinTemp'].between(A, B)
for i in range(len(mask)):
    if mask[i] == False:
        raindf = raindf.drop([i], axis=0)    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




