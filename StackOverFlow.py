#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# In this project we will do **exploratory data analysis** about the stackOverflow dataset 2022. Stack Overflow is a question and answer website for professional and enthusiast programmers.As of March 2021 Stack Overflow has over 14 million registered users,and has received over 21 million questions and 31 million answers.The site and similar programming question and answer sites have globally mostly replaced programming books for day-to-day programming reference in the 2000s, and today are an important part of computer programming.Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: JavaScript, Java, C#, PHP, Android, Python, jQuery, and HTML.
# 
# **Things We will do as follow:**
# 
# - Data Cleaning
# - Data Manipulation
# - Data Visualization

# In[1]:


# Let's import the libraries
import numpy as np                  #For performing mathematical calculation
import pandas as pd                 #It is used to analyze data.
import matplotlib
import matplotlib.pyplot as plt     #Matplotlib as seaborn, both we will use for the visualization
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")   #Just to avoid warnings
import statistics as stat
from scipy.stats import kurtosis,skew


# In[2]:


sns.set_style("darkgrid")

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10,6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"


# In[3]:


# Import the dataset
df = pd.read_csv("survey_results_public 2022.csv")
df_schema = pd.read_csv("survey_results_schema 2022.csv")


# In[4]:


#preview the public dataset
df.head()


# In[5]:


#preview the schema dataset
df_schema.head()


# In[6]:


#check the number of rows and columns
df_schema.shape


# In[7]:


# view the shape
print("Public dataset has total Rows of ",df.shape[0])
print("Public dataset has total Columns of ",df.shape[1])
print("Schema dataset has total Rows of ",df_schema.shape[0])
print("Schema dataset has total Columns of ",df_schema.shape[1])


# In[8]:


# Let's see the size of the dataset
df.info()


# - From the above table we can say that public dataset has lots of missing values.
# - There are 79 total columns  in which only 6 are numeric columns and rest are categorical.
# - it usage memory of 44 MB
# - Later We will minimize its memory usage by the Optimizing Data Types methods.

# #### Data type and missing value
# 
# Public dataset contains lots of columns so it is easier to make a table of entire columns and analysis datatypes, missing value and number of unique value in each columns.

# In[9]:


df_type_missing = pd.DataFrame({"Pandas_dtype":df.dtypes.values,
                               "Missing_value":df.isnull().sum().values,
                               "% Missing_value":df.isnull().sum() * 100 / len(df),
                               "Unique_value":[df[col].nunique() for col in df.columns]})


# In[10]:


#preview the dataframe
df_type_missing.head()


# Now we perfome optimizing data types to minimize its memory size.So first of all let's deal with numeric column.

# In[11]:


numeric_cols = df_type_missing[df_type_missing["Pandas_dtype"] != "object"]
numeric_cols


# Let's check minimum and maximum value of datatypes in pandas:

# In[12]:


#check integers only
print("Maximum value of int8 type is: ",np.iinfo(np.int8).max)
print("Minimum value of int8 type is: ",np.iinfo(np.int8).min)
print("Maximum value of int16 type is: ",np.iinfo(np.int16).max)
print("Minimum value of int16 type is: ",np.iinfo(np.int16).min)
print("Maximum value of int32 type is: ",np.iinfo(np.int32).max)
print("Minimum value of int32 type is: ",np.iinfo(np.int32).min)
print("Maximum value of int64 type is: ",np.iinfo(np.int64).max)
print("Minimum value of int64 type is: ",np.iinfo(np.int64).min)


# In[13]:


#check integers only
print("Maximum value of float16 type is: ",np.finfo(np.float16).max)
print("Minimum value of float16 type is: ",np.finfo(np.float16).min)
print("Maximum value of float32 type is: ",np.finfo(np.float32).max)
print("Minimum value of float32 type is: ",np.finfo(np.float32).min)
print("Maximum value of float64 type is: ",np.finfo(np.float64).max)
print("Minimum value of float64 type is: ",np.finfo(np.float64).min)


# Now we got the data range of each datatype. Let's change the datatype of each numeric columns to float32.

# In[14]:


numeric_list = list(numeric_cols.index)
df[numeric_list]  =df[numeric_list].astype("float32")


# Now deal with the categorical columns

# In[15]:


categoric_cols = df_type_missing[(df_type_missing["Pandas_dtype"] == "object") & (df_type_missing["Unique_value"] <= 10)]
categoric_cols


# Let's make above columns into categorical because these have minimum number of unique value.

# In[16]:


categorical_list = list(categoric_cols.index)
df[categorical_list] = df[categorical_list].astype("category")


# In[17]:


# Checking the memorysize
df.info()


# We reduce 15 MB by just changing the data type. There are several methods by which we can reduce the momery size.For example:
# 1. Changing file format
# 2. Read only required columns
# 3. Read data in chunks etc.

# Let's drop those columns which has more 50% of missing data.

# In[18]:


missing_value_df = df_type_missing[df_type_missing["% Missing_value"]> 50]
missing_value_list = list(missing_value_df.index)
df.drop(missing_value_list,axis=1,inplace=True)


# In[19]:


#Check the rows and columns in the dataset
df.shape


# In[20]:


#Check the column which does not contain null values
no_missing_value = df_type_missing[df_type_missing["% Missing_value"]== 0]
no_missing_value


# - `ResponseId` is does not contain null values and it has only unique id.
# - `MainBranch` has only 6 unique values.

# The following table provides statistical information in descriptive analysis.

# In[21]:


numeric_cols = list(df.select_dtypes(np.number).columns)
statistical_df=pd.DataFrame(data={"Max":[df[col].max() for col in  numeric_cols],
                 "Range":[(df[col].max()-df[col].min()) for col in numeric_cols],
                 "IQR": [(df[col].quantile(0.75)-df[col].quantile(0.25)) for col in numeric_cols],
                  "Mode":[stat.mode(df[col]) for col in numeric_cols],
                  "Mad": [df[col].mad() for col in numeric_cols],
                  "Kurtosis":[kurtosis(df[col],fisher=False) for col in numeric_cols],
                  "Skewness":[skew(df[col]) for col in numeric_cols],
                  "Mean":[df[col].mean() for col in numeric_cols],
                  "Std": [df[col].std() for col in numeric_cols],
                  "Min": [df[col].min() for col in numeric_cols],
                  "25%": [df[col].quantile(0.25) for col in numeric_cols],
                  "50%": [df[col].quantile(0.50) for col in numeric_cols],
                  "75%": [df[col].quantile(0.75) for col in numeric_cols]},
                            index= numeric_cols)


# In[22]:


statistical_df


# Let's understand each and every columns in the dataset. To do this let's deal with schema table because in this table describes the detail about the column's purpose.

# In[23]:


df_schema.head()


# Drop the unnecessery columns such as `qid`,`force_resp`,`type`,`selector`

# In[24]:


df_schema.drop(["qid","force_resp","type","selector"],axis=1,inplace=True)


# In[25]:


# Drop 1st 3 rows 
df_schema.drop(df_schema.head(3).index,inplace=True)


# In[26]:


#Rename the columns
df_schema.columns = ["Feature_name","Feature_detail"]


# In[27]:


pd.options.display.max_colwidth = 200
df_schema.head()


# Feature detail contains html tag so to remove this we need to import re library. This library helps us to remove html tag.

# In[28]:


import re
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')  #Symbols within which html tags are usually written


# In[29]:


def cleanhtml(text):
    cleantext = re.sub(CLEANR, '', text)
    return cleantext


# In[30]:


df_schema["Feature_detail"] = df_schema["Feature_detail"].apply(lambda x: cleanhtml(x))


# In[31]:


df_schema.head()


# Now the schema dataset has been ready. Let's explore each and every columns in public dataset.

# In[32]:


df.columns


# **MainBranch**
# 
# - This columns contains profession.Such as:
# - Developer: who writes code
# - Learner: Who learns code
# - CoderForWork: Not a devloper but code as part of work
# - CoderAsHobby: Coding as a hobby
# - Retaired_developer: Used to a developer but now no longer a developer
# - NoneOfThese: Not mentioned

# In[33]:


df["MainBranch"].value_counts()


# In[34]:


# Lets change the profession into smaller version
name_change = {"I am a developer by profession":"Developer",
              "I am learning to code":"Learner",
              "I am not primarily a developer, but I write code sometimes as part of my work":"CoderForWork",
              "None of these":"NoneOfThese",
              "I used to be a developer by profession, but no longer am":"Retaired_developer"}
df["MainBranch"] = df["MainBranch"].replace(name_change)


# In[35]:


main_branch = df["MainBranch"].value_counts()


# In[36]:


plt.pie(main_branch.values,labels=main_branch.index,autopct='%1.1f%%')
plt.title("Types of MainBranch")
plt.show()


# From above graph, it has been seen that the majority coders are developer. 

# **Employment**

# In this feature we will learn about the employment status such as weather people are full time employee or part time or free lancer or we can say that weather they are employed or student etc.

# In[37]:


df["Employment"].value_counts()


# It has been difficult to read the above result. So let's split this column and try to read every Feature.

# In[38]:


employment = df["Employment"].str.split(",",expand=True).add_prefix("Employment_")
employment


# In[39]:


features = list(employment.columns)
for feature in features:
    print("{} has {} number of unique value.".format(feature,employment[feature].nunique()))
    print()


# In[40]:


# Let's check the frequency for every element
def unique_values(col):
    return employment[col].value_counts().sum()


# In[41]:


for col in features:
    print(unique_values(col))


# Except `Employement_0` and `Employement_1`, other feature have lots of null values so let's not consider these feature.

# In[42]:


#Let's add these two columns into public dataset
df["Employment"] = employment["Employment_0"]
df["EmploymentCategory"] = employment["Employment_1"]


# In[43]:


df.head()


# Let's explore the newly added columns:

# In[44]:


employment["Employment_0"].value_counts()  #Frequency for every element


# In[45]:


employment["Employment_0"].isnull().sum()  #Look into null values


# This column contains 1559 null values. Let's compare this column with mainbranch so that we can find how can we impute these null values.

# In[46]:


df[df["Employment"].isnull()]    #Check dataframe by selecting only null values in Employment


# It has been seen that in the employement dataframe total null values is 1559 and in the mainbrach column NoneOfThese is 1497. So overall values are comparable.So let's impute null values accroding to mainbranch.

# In[47]:


df.loc[(df["MainBranch"]=="NoneOfThese") & (df["Employment"].isnull()),"Employment"] = "NoneOfThese"
df.loc[(df["MainBranch"]=="Developer") & (df["Employment"].isnull()),"Employment"] = "Employed"
df.loc[(df["MainBranch"]=="Learner") & (df["Employment"].isnull()),"Employment"] = "Student"
df.loc[(df["MainBranch"]=="CoderForWork") & (df["Employment"].isnull()),"Employment"] = "Employed"
df.loc[(df["MainBranch"]=="CoderAsHobby") & (df["Employment"].isnull()),"Employment"] = "Not employed"
df.loc[(df["MainBranch"]=="Retaired_developer") & (df["Employment"].isnull()),"Employment"] = "Retired"
df.loc[df["Employment"].isnull(),"Employment"] = "NoneOfThese"


# In[48]:


# Let's check null values
df["Employment"].isnull().sum()


# In[49]:


plt.figure(figsize=(13,6))
sns.countplot(x=df["Employment"])
plt.title("Employment Distribution")
plt.show()


# From the above graph, it has been seen that most of the people are employed then student.

# Now deal with the EmploymentCategory columns whether full time, part time, self independent etc.

# In[50]:


#check frequency of every element
df["EmploymentCategory"].value_counts()


# In[51]:


#look into null values
df["EmploymentCategory"].isnull().sum()


# In[52]:


df[df["EmploymentCategory"] =="full-time"]


# I didn't get any data by filtering with `full-time` because this column is created by split method so before any element there is gap available so I have to put gap before the element to get the data.

# In[53]:


df[df["EmploymentCategory"] ==" full-time"].head()


# Here we need to strip the EmployementCategoryColumn then we can see result.

# In[54]:


df["EmploymentCategory"] = df["EmploymentCategory"].str.strip()


# In[55]:


df[df["EmploymentCategory"] =="full-time"].head(1)


# In[56]:


df["EmploymentCategory"].value_counts()


# In[57]:


# first change the name of the element
change_name = {"full-time":"Full Time",
              "freelancer":"Freelancer",
              "full-time;Independent contractor":"Full Time",
              "full-time;Student":"Full Time",
              "part-time":"Part Time",
              "but looking for work":"Not Working",
              "full-time;Employed":"Full Time",
              "full-time;Not employed":"Full Time",
              "part-time;Employed":"Part Time",
              "and not looking for work":"Not Working",
              "part-time;Not employed":"Part Time",
              "part-time;Independent contractor":"Freelancer",
              "but looking for work;Independent contractor":"Freelancer",
              "but looking for work;Not employed":"Not Working",
              "and not looking for work;Retired":"Not Working",
              "part-time;Retired":"Part Time",
              "but looking for work;Employed":"Not Working",
              "full-time;Retired":"Full Time",
              "but looking for work;Retired":"Not Working"}


# In[58]:


df["EmploymentCategory"] = df["EmploymentCategory"].replace(change_name)


# Let's deal with null values. Will do everything same as we did for employment column

# In[59]:


df.loc[(df["Employment"]=="NoneOfThese") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Not mentioned"
df.loc[(df["Employment"]=="Employed") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Full Time"
df.loc[(df["Employment"]=="Student") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Part Time"
df.loc[(df["Employment"]=="Independent contractor") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Freelancer"
df.loc[(df["Employment"]=="Not employed") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Not mentioned"
df.loc[(df["Employment"]=="Retired") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "Not mentioned"
df.loc[(df["Employment"]=="I prefer not to say") & (df["EmploymentCategory"].isnull()),"EmploymentCategory"] = "NoneOfThese"
df.loc[df["EmploymentCategory"].isnull(),"EmploymentCategory"] = "NoneOfThese" 


# In[60]:


df["EmploymentCategory"].isnull().sum()


# In[61]:


sns.countplot(df["EmploymentCategory"])
plt.show()


# **RemoteWork**

# This feature describes the work situation such as:
# - fully remote
# - Full in-person
# - Hybrid (some remote, some in-person)

# In[62]:


df["RemoteWork"].value_counts()  #Look into frequency of every element


# In[63]:


df["RemoteWork"].isnull().sum() #Look into null values


# Let's impute null values for this column by comparing with mainbranch and employment column.

# In[64]:


df.loc[(df["MainBranch"]=="Developer") & (df["Employment"]=="Employed") & (df["RemoteWork"].isnull()),"RemoteWork"] = "Fully remote"


# In[65]:


df["RemoteWork"] = df["RemoteWork"].astype("object")


# In[66]:


df.loc[df["RemoteWork"].isnull(),"RemoteWork"] = "NoneOfThese"


# In[67]:


change_value = {"Hybrid (some remote, some in-person)":"Hybrid"}
df["RemoteWork"] = df["RemoteWork"].replace(change_value)          


# In[68]:


df["RemoteWork"].value_counts()


# In[69]:


df["RemoteWork"].isnull().sum()


# In[70]:


ax = df['RemoteWork'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Distributionof RemoteWork")
ax.set_xlabel("RemoteWork")
ax.set_ylabel("Frequency")
plt.show()


# **CodingActivities**

# This feature desccribe the coding activities means treating coding as a hobby or doing coding for work purpose or other purpose.

# In[71]:


df["CodingActivities"].value_counts()  #Look into frequency of every element


# In[72]:


df["CodingActivities"].isnull().sum()


# Let's impute null values for this column by comparing with mainbranch and employment column

# In[73]:


df.loc[(df["Employment"]=="NoneOfThese") & (df["CodingActivities"].isnull()),"CodingActivities"] = "NoneOfThese"
df.loc[(df["RemoteWork"]=="NoneOfThese") & (df["CodingActivities"].isnull()),"CodingActivities"] = "NoneOfThese"
df.loc[df["CodingActivities"].isnull(),"CodingActivities"] = "Hobby"


# In[74]:


df.loc[df["CodingActivities"].str.contains("Hobby"),"CodingActivities"] = "Hobby"
df.loc[df["CodingActivities"].str.contains("I don’t"),"CodingActivities"] = "Not a Hobby"
df.loc[df["CodingActivities"].str.contains("Freelance"),"CodingActivities"] = "Freelance"
df.loc[df["CodingActivities"].str.contains("Contribute to open-source projects"),"CodingActivities"] = "Contribute to open-source projects"
df.loc[df["CodingActivities"].str.contains("Bootstrapping a business"),"CodingActivities"] = "Bootstrapping a business"
df.loc[df["CodingActivities"].str.contains("Other"),"CodingActivities"] = "Other"


# In[75]:


df["CodingActivities"].isnull().sum()


# In[76]:


df["CodingActivities"].value_counts().values


# In[77]:


plt.pie(df["CodingActivities"].value_counts().values,labels=df["CodingActivities"].value_counts().index,autopct="%1.1f%%")
circle = plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.show()


# Most of the person treat coding as hobby.

# **EdLevel**

# This feature describes the highest level of formal education of the users.

# In[78]:


df["EdLevel"].value_counts()  #Look into frequency of every element


# In[79]:


df["EdLevel"].isnull().sum()


# In[80]:


df["EdLevel"] = df["EdLevel"].astype("object")


# In[81]:


df.loc[(df["RemoteWork"]=="NoneOfThese") & (df["EdLevel"].isnull()),"EdLevel"] = "NoneOfThese"
df.loc[df["EdLevel"].isnull(),"EdLevel"] = "Something else"


# In[82]:


df.loc[df["EdLevel"].str.contains("Bachelor’s degree"),"EdLevel"] = "Bachelor’s degree"
df.loc[df["EdLevel"].str.contains("Master’s degree"),"EdLevel"] = "Master’s degree"
df.loc[df["EdLevel"].str.contains("without earning"),"EdLevel"] = "Without degree"
df.loc[df["EdLevel"].str.contains("Secondary school"),"EdLevel"] = "Secondary school"
df.loc[df["EdLevel"].str.contains("Associate degree"),"EdLevel"] = "Associate degree"
df.loc[df["EdLevel"].str.contains("Other doctoral degree"),"EdLevel"] = "Other doctoral degree"
df.loc[df["EdLevel"].str.contains("Professional degree"),"EdLevel"] = "Professional degree"


# In[83]:


df["EdLevel"].value_counts()


# In[84]:


y=df["EdLevel"].value_counts().index
x=df["EdLevel"].value_counts().values

plt.barh(y,x)
plt.xlabel("Count")
plt.ylabel("Education Level")  
plt.title("Distribution of Education Level")
plt.show()


# Most of the coders completed their bachelor's degree.

# **LearnCode**

# This feature describe source of learning code.
# - Here school includes University, college etc.

# In[85]:


df["LearnCode"].value_counts()  #Look into frequency of every element


# In[86]:


df["LearnCode"].nunique()


# In[87]:


df["LearnCode"].isnull().sum()


# In[88]:


df.loc[(df["EdLevel"]=="NoneOfThese") & (df["LearnCode"].isnull()),"LearnCode"] = "NoneOfThese"
df.loc[(df["EdLevel"]=="Something else") & (df["LearnCode"].isnull()),"LearnCode"] = "Not Mentioned"
df.loc[df["LearnCode"].isnull(),"LearnCode"] = "Books or online resources"


# In[89]:


df.loc[df["LearnCode"].str.contains("School"),"LearnCode"] = "School"
df.loc[df["LearnCode"].str.contains("Other online resources"),"LearnCode"] = "Online resources"
df.loc[df["LearnCode"].str.contains("Books"),"LearnCode"] = "Books or online resources"
df.loc[df["LearnCode"].str.contains("Hackathons"),"LearnCode"] = "Hackathons"
df.loc[df["LearnCode"].str.contains("Friend or family member"),"LearnCode"] = "Friend or family member"
df.loc[df["LearnCode"].str.contains("On the job training"),"LearnCode"] = "On the job training"
df.loc[df["LearnCode"].str.contains("Certification"),"LearnCode"] = "Online Courses or Certification"
df.loc[df["LearnCode"].str.contains("Coding Bootcamp"),"LearnCode"] = "Coding Bootcamp"
df.loc[df["LearnCode"].str.contains("Other"),"LearnCode"] = "Other"


# In[90]:


df["LearnCode"].value_counts().sort_values()


# In[91]:


y=df["LearnCode"].value_counts().index
x=df["LearnCode"].value_counts().sort_values()

plt.barh(y,x,color="r")
plt.xlabel("Count")
plt.ylabel("Education Level")  
plt.title("Distribution of Source of Learning Code")
plt.show()


# **LearnCodeOnline**

# This feature decribes about resources from which user learn code.

# In[92]:


df["LearnCodeOnline"].value_counts()  #Look into frequency of every element


# In[93]:


df["LearnCodeOnline"].nunique()


# In[94]:


df["LearnCodeOnline"].isnull().sum()


# In[95]:


df.loc[(df["RemoteWork"]=="NoneOfThese") & (df["LearnCodeOnline"].isnull()),"LearnCodeOnline"] = "NoneOfThese"
df.loc[(df["LearnCode"]=="Not Mentioned") & (df["LearnCodeOnline"].isnull()),"LearnCodeOnline"] = "Not Mentioned"
df.loc[df["LearnCodeOnline"].isnull(),"LearnCodeOnline"] = "Not Mentioned"


# In[96]:


df.loc[df["LearnCodeOnline"].str.contains("Technical documentation"),"LearnCodeOnline"] = "Technical documentation"
df.loc[df["LearnCodeOnline"].str.contains("Stack Overflow"),"LearnCodeOnline"] = "Stack Overflow"
df.loc[df["LearnCodeOnline"].str.contains("Programming Games"),"LearnCodeOnline"] = "Programming Games"
df.loc[df["LearnCodeOnline"].str.contains("Blogs"),"LearnCodeOnline"] = "Blogs"
df.loc[df["LearnCodeOnline"].str.contains("Written Tutorials"),"LearnCodeOnline"] = "Written Tutorials"
df.loc[df["LearnCodeOnline"].str.contains("Online books"),"LearnCodeOnline"] = "Online books"
df.loc[df["LearnCodeOnline"].str.contains("Video-based Online Courses"),"LearnCodeOnline"] = "Video-based Online Courses"
df.loc[df["LearnCodeOnline"].str.contains("How-to videos"),"LearnCodeOnline"] = "How-to videos"
df.loc[df["LearnCodeOnline"].str.contains("Online challenges"),"LearnCodeOnline"] = "Online challenges"
df.loc[df["LearnCodeOnline"].str.contains("Online forum"),"LearnCodeOnline"] = "Online forum"
df.loc[df["LearnCodeOnline"].str.contains("Written-based Online Courses"),"LearnCodeOnline"] = "Written-based Online Courses"
df.loc[df["LearnCodeOnline"].str.contains("Coding sessions"),"LearnCodeOnline"] = "Coding sessions"
df.loc[df["LearnCodeOnline"].str.contains("Other"),"LearnCodeOnline"] = "Other"
df.loc[df["LearnCodeOnline"].str.contains("Interactive tutorial"),"LearnCodeOnline"] = "Interactive tutorial"


# In[97]:


df["LearnCodeOnline"].value_counts()


# In[98]:


y=df["LearnCodeOnline"].value_counts().index
x=df["LearnCodeOnline"].value_counts().sort_values()

plt.barh(y,x,color="c")
plt.xlabel("Count")
plt.ylabel("Education Level")  
plt.title("Distribution of Resource of Learning Code")
plt.show()


# **YearsCode**

# It describes the how many years user are doing coding.

# In[ ]:





# In[ ]:





# In[ ]:




