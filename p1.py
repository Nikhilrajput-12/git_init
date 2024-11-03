import numpy as np
import pandas as pd

# basic data structure in pandas
# 1.series = 1d labelled array holding data of any type such as integers,strings,python objects etc
# 2.dataframe:2d data structure tha hold data like a two dimenson array or a table with rows and collumns



#object creation
# creating a series by passing a list value, letting pandas create a default RANGEINDEX
s=pd.Series([1,3,5,np.nan,6,8])
# print(s)

# creating a dataframe by Passing numpy array with a datetime index using date_range()
# and lablledd columns

'''dates=pd.date_range("20130101",periods=6)
print(dates)'''


'''df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)
'''
#creating a dataframe by using dictionary of objects where the keys are the column
# labels and the values are the column values

import numpy as np

# df2 = pd.DataFrame(
#     {
#         "A": 1.0,  # Broadcasts to length 4
#         "B": pd.Timestamp("20130102"),  # Broadcasts to length 4
#         "C": pd.Series(1, index=list(range(4)), dtype="float32"),  # Length 4
#         "D": np.array([3] * 4, dtype="int32"),  # Length 4
#         "E": pd.Categorical(["test", "train", "train", "test"]),  # Now length 4
#         "F": "foo"  # Broadcasts to length 4
#     }
# )

# print(df2.dtypes)
# print(df2.A)
# print(df2.abs)

#---> VIEWING DATA
#USE   DataFrame,head() and DataFrame.tail() to view the top and bottom rows of the frame respectively:

# print(df.head())
# print(df.tail())


# display the Dataframe.index or Dataframe.columns:
'''i=pd.DataFrame(
    {
    "A" : [1,2,3,4],
    "B":[5,6,7,8]
    },index=["row1","row2","row3","row4"]
)
print(i)
print(i.index)
print(i.columns)'''


# If you call df.to_numpy(), the output will be a NumPy array containing the DataFrame’s values:
'''n =pd.DataFrame(
    {
    "A" : [1,2,3,4],
    "B":[5,6,7,8]
})
numpy_array=n.to_numpy()
print(numpy_array)
'''
# describe() shows a quick statistic summary of your data:
'''ex =pd.DataFrame(
    {
    "A" : [1,2,3,4],
    "B":[5,6,7,8]
})
print(ex.describe())
'''


# Transposing your data:
'''t=pd.DataFrame(
    {
        "A":[1,2,3],
        "B":[4,5,6]
    }
)
print(t)
print(t.T)'''


#DataFrame.sort_index()  sort by an axis:
# DataFrame.sort_index() is a method in pandas that allows you to sort the rows or columns of a DataFrame based on its index (row labels) or columns (column labels). This can be particularly useful when you need to reorder your data for better analysis or presentation.



# Create a DataFrame with a custom index
'''df = pd.DataFrame({
    "c": [4, 3, 2, 1],
    "B": [8, 7, 6, 5]
}, index=["row2", "row3", "row1", "row4"])

print("Original DataFrame:")
print(df)

# Sort the DataFrame by index (row labels)
sorted_df = df.sort_index(axis=0)  #sorted by row
sorted_df1 = df.sort_index(axis=1)  #sorted by column

print("\nSorted DataFrame by index:")
print(sorted_df)
print("\n",sorted_df1)

'''


# DataFrame.sort_values()
'''import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "A": [4, 3, 2, 1],
    "B": [8, 7, 6, 5]
})

print("Original DataFrame:")
print(df)

# Sort the DataFrame by column 'A' by default sort in ascending order
sorted_df = df.sort_values(by='A')
sorted_df1 = df.sort_values(by='A',ascending=False) #descending order sort



print("\nSorted DataFrame by column 'A':")
print(sorted_df)
print("\n",sorted_df1)'''
# sort_index() is used to sort by index labels (either row or column labels).
# sort_values() is used to sort by the actual data values (either column or row values).


# Sorting by Multiple Columns:


'''df_multi=pd.DataFrame({
    "A":[1,2,3,1],
    "B":[3,2,1,2]
})
print(df_multi)
sorted_multi=df_multi.sort_values(by=['A','B'],ascending=[True,True])
print("\n",sorted_multi)'''


#---> GETITEM
# (column select)
# n pandas, the __getitem__ method (i.e., []) is used to access columns in a DataFrame. Here’s how it works:
# When you pass a single label to the DataFrame using df['A'], it selects the column labeled A and returns it as a Series.

'''data ={'A' :[1,2,3],'B':[4,5,6]}
print(data)
df=pd.DataFrame(data)
# print(df)

col_A =df['A']   #select column A
# print(col_A)

cols_AB=df[['A','B']]
# print(cols_AB)  #if you want select multiple column

df['A'] returns a Series.
df[['A', 'B']] returns a DataFrame.

#for a dataframe,passing a slice : selects matching rows:

slice=df[0:3]  #slicing the 1st 3 rows
print(slice)'''


# selection by label  (row select)
# using  dataframe.loc() or dataframe.at()

# Generating a sample date range
'''dates =pd.date_range('20230101',periods=5)

# Creating a DataFrame with dates as index
df=pd.DataFrame(np.random.randn(5,4),index=dates,columns=list('AbCd'))
print(df)
# 
# Accessing the row corresponding to the first date
first_row =df.loc[dates[0]]
print(first_row)


# Selecting all rows (:) with a select column labels:
multi_row=df.loc[:,['A','b']]
print(multi_row)


# for label slicing,both endpoint are include:
sel=df.loc["20230103":"20230104",["A","b"]]
print(sel)

# Selecting a single row and column label returns a scalar:
single=df.loc[dates[0],"A"]
print(single)

# For getting fast access to a scalar (equivalent to the prior method):
# The df.at[] function in pandas is a specialized method used for accessing a single scalar value from the DataFrame. It's designed to be faster than df.loc[] when you are accessing a single element because it's optimized specifically for this use case.
fast=df.at[dates[0],"A"]
print(fast)'''


#SELECTION BY POSITION
'''
# DataFrame.iloc() or DataFrame.iat()

pos=df.iloc[2]  #selecting the row at index 2
print(pos)

#intger slices acts similar to numpy/python
pi=df.iloc[2:4,0:2]  #2:4 rows and 0:2 columns slect
print(pi)

# list of inetger position location:
p=df.iloc[[1,2,3],[0,2]]  #[1,2,3] is a rows that we select and [0,2] specify the column that we select
print(p)

# for slicing rows explicity
all_col=df.iloc[1:3,:]   # here we select row 1:3 and we select all column by usng label(:)
print(all_col)

all_row=df.iloc[:,1:3]
print(all_row)

explicit_value=df.iloc[1,1]
print(explicit_value)

# For getting fast access to a scalar (equivalent to the prior method):
at=df.iat[1,1]
print(at)'''



#---> BOOLEAN INDEXING

dates = pd.date_range("20130101", periods=6)
# print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
# print(df)

# Select rows where df.A is greater than 0.
'''filtered_df=df[df["A"]>0]  #filter  A

f=df[(df["A"]>0) & (df["B"]>0)] #filter A and B

print(filtered_df)
print(f)

# Selecting values from a DataFrame where a boolean condition is met:

values_df=df[df>0]
print(values_df)

# using   isin()  method for filtering

df2 =df.copy()
print(df2)

df2["E"] =["one","one","two","three","four","three"]  #create a new column E with data
print(df2)

is_in =df2[df2["E"].isin(["two","four"])]
print(is_in)'''



#---> SETTING
# Setting a new column automatically aligns the data by the indexes:


s1=pd.Series([1,2,3,4,5,6],index=pd.date_range("20130102",periods=6))
# print(s1)

df["F"] =s1
# print(df)

# Setting values by label:
df.at[dates[0],"A"] =0

# Setting values by position:
df.iat[0,1]=0    #0row,1column

# Setting by assigning with a NumPy array:
df.loc[:,"D"] =np.array([5]*len(df))
# print(df)

# A where operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
# print(df2)



#IMPORTANT
#  --> MISSING DATA
#1.Dataframe.dropna()   #remove missing data
#2.Dataframe.fillna()   #fill missing data
#3.isna()               #fill with boolean value

# For NumPy data types, np.nan represents missing data. It is by default not included in computations. See the Missing Data section.
# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data:

df1=df.reindex(index=dates[0:4], columns=list(df.columns)+ ["E"])  # Reindex df to select first four dates and add column "E"
df1.loc[dates[0] : dates[1],"E"] =1    # Set column "E" to 1 for specific date range
print(df1)

# imp--> DataFrame.dropna() drops any rows that have missing data:
df1_cleaned=df1.dropna(how="any")
print(df1_cleaned)      #delete the rows where NAN value is avilabale

#Dataframe.fillna() fills missing data
df1.fillna(value=5)  #fill NAN value
print(df1)

boolean=pd.isna(df1)    #they fill TRUE where NAN value is available
print(boolean)

