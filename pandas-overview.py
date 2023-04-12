#!/usr/bin/env python
# coding: utf-8

# In[9]:


# why pandas? gives more flexibility and allows to work with larger datasets
# groupby
# getdummy
# label encoder 1 hot encoding 


import pandas as pd

# Loading data into pandas
poke=pd.read_csv('Pokemon.csv')
# pd.read_excel('file name')

print(poke.head(5))#to show the upper 5 rows
print('----------------------------------------------------------------------------------------------')
print(poke.tail(5))#to print the last 5 rows in 


# In[43]:


# Reading data in pandas

#reading the headers
print(poke.columns)
print('----------------------------------------------------------------------------')

#reading each column
print(poke['Name'])
# for i in range(5):
#     print(poke['Name'][i],poke['Type 1'][i])

print('-#_#_#_#_#_#_#_#_#_#_#_#_#-')
# another way to print multiple columns
print(poke[['Name','Type 1','HP']])
print('----------------------------------------------------------------------------')



#reading each row
print(poke.head(1))
print()
print(poke.iloc[0:4]) #iloc---integer location

print()
print('--------getting specific rows and columns-------')
for index,rows in poke.iterrows():
    print(index,rows['Name'])
print()
print('--------filtering the data according to the specification ''Fire''---------')
print(poke.loc[poke['Type 1']=='Fire'])


print('----------------------------------------------------------------------------')
#reading a specific loaction
print(poke.iloc[2,1]) #here iloc[2,1]-->iloc[Row,Column]


# In[52]:


#sorting and describing data
poke.describe() # gives all sort of data like mean ,std,max min ,25%, 75%

poke.sort_values(['Name','HP'],ascending=False) #it will sort by name


# #making changes in data
# poke.head()
# 
# #adding a column to add all the different stats
# print('way 1--->')
# poke['Total']=poke['HP']+poke['Attack']+poke['Defense']+poke['Sp. Atk']+poke['Sp. Def']+poke['Speed']
# poke.head()
# 
# 
# #dropping a column
# poke=poke.drop(columns=['Total'])
# poke.head()
# 
# print('way 2--->')
# poke['total']=poke.iloc[:,4:10,sum(axis=1)]
# poke.head()

# In[68]:


#making changes in data
poke.head()

#adding a column to add all the elements stats
print('way--1')
poke['Total']=poke['HP']+poke['Attack']+poke['Defense']+poke['Sp. Atk']+poke['Sp. Def']+poke['Speed']
poke.head()

#dropping the total column
poke=poke.drop(columns=['Total'])
poke.head()

print('way--2')
poke['Total']=poke.iloc[:,4:10].sum(axis=1)#iloc takes rows and columns.. : means all rows,4:10 from 4th column to 9th
poke.head()


# In[72]:


#rearranging the columns
cols=poke.columns.values
print(cols)

poke=poke[cols[0:4] +[cols[-1]]+cols[4:12]]


# In[73]:


# saving modified data
poke.to_csv('modified.csv') #if there are index then used poke.to_cv('modified.csv,index=False')


# In[87]:


#filtering data
new_poke=poke.loc[(poke['Type 1']=='Grass') & (poke['Type 2']=='Poison')]

# new_poke.to_csv('filtered.csv')

#drop=true get rids of the old indecies
# new_poke=new_poke.reset_index(drop=True)
#to save in the same dataframe
new_poke.reset_index(drop=True,inplace=True)
new_poke

#filtering data containing the word mega
poke.loc[poke['Name'].str.contains('Mega')]


# In[88]:


#filtering the data not containing the word mega
poke.loc[~poke['Name'].str.contains('Mega')]
poke


# In[98]:


#using regex to filtering
import re
poke.loc[poke['Type 1'].str.contains('Fire | Grass',regex=True)]


# In[99]:


#using regex part 2
poke.loc[poke['Name'].str.contains('^pi[a-z]*',flags=re.I,regex=True)]


# In[106]:


##conditional changes
#changes the name from fire to blaze
poke.loc[poke['Type 1']=='Fire','Type 1']='Blaze'
#poke
poke.loc[poke['Type 1'].str.contains('Blaze')]


# In[118]:


#aggregate Statistics (Groupby)
# find the mean and then sort them by Defense values in descending order
poke.groupby(['Type 1']).mean().sort_values('Attack',ascending=False)
poke['count']=1
# poke.groupby(['Type 1']).sum().sort_values('Attack')
# poke.groupby(['Type 1']).count()
# poke.groupby(['Type 1']).count()['count']
#grouping multiple parameters
poke.groupby(['Type 1','Type 2']).count()['count']


# In[119]:


#working with large amount of data
new_df=pd.DataFrame(columns=poke.columns)
#chunksize 5 means 5 rows
for poke in pd.read_csv('filtered.csv',chunksize=5):
    results=poke.groupby(['Type 1']).count()
    
    new_df=pd.concat([new_df,results])
    


# In[ ]:




