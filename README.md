# Starbucks Capstone
## This project is part of the Udacity Data Science Nanodegree

![Starbucks](https://stories.starbucks.com/uploads/2020/06/Starbucks-Virtual-Backgrounds-Olive-Way-Seattle.jpg)

---

## 👨‍💼🤝👨‍💼 Business Understanding Intro
Data sets contain simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to mobile app users. An offer can be merely an advertisement for a drink or an offer such as a discount or BOGO (buy one get one free). Some users might not receive any offers during certain weeks. 

Not all users receive the same offer, which is a challenge to solve with this data set.

Every offer has a validity period before the offer expires. For example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

We are given transactional data showing user purchases made on the app, including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record of each offer a user receives and a record of when a user views the offer. There are also records for when a user completes an offer. 

Lastly, it is essential to note that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, this data set has a few things to watch out for. Customers do not opt into the offers they receive; in other words, a user can receive an offer, never actually view the offer, and still complete it. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer. -- ***We will clean the data and apply simple feature engineering to accomplish this***

---
# 👨‍💻 Data Understanding

The data is contained in 3 files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed


**Schema and explanation of each variable in the files:**

`portfolio.json`
* id (string) - offer id 
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

`profile.json`
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

`transcript.json`
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record
---
# 🥅 Business Goal

Find out which features are most important in predicting a successful offer so business analysts can work with the marketing team to better target customers.

How will this be accomplished?

- Exploratory Data Analysis (EDA): create visuals that may prove helpful to the business analysts and marketing teams 

- Clean and merge the data sets into a single data set

- Split the data for training and testing and apply feature scaling

- Train 3 machine learning models on the data and optimize the "best" model with parameter tuning to see if the performance metric (F1-Score) may improve

- Visualize the selected models **a)** Feature Importance, **b)** ROC Plot, and **c)** Classification Report

---
# 🧹Data Preparation
```python
# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
```

Here is a little function that I use to save some time on basic EDA:

```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def data_intro(data):
    print("\033[4mFirst 5 rows of data:\033[0m", '\n'*2, data.head(), '\n'*2)
    print("\033[4mData Info:\033[0m", '\n')
    print(data.info(),'\n'*2)
    print("\033[4mBasic statistics for numeric features:\033[0m", '\n'*2, data.describe(include=np.number), '\n'*2)
    print("\033[4mNULL value examination (%):\033[0m", '\n'*2, round(100 * data.isnull().sum()/len(data), 2), '\n'*2)
    print("\033[4mBarchart of existing values (Full bar indicates no missing values):\033[0m",'\n')
    plt.title('Barchart of present values (Full bar indicates no missing values)', fontsize=30)
    plt.xlabel('Features', fontsize=25)
    plt.ylabel('%', fontsize=25)
    msno.bar(data)
    plt.show()
    print('\n')
    print("\033[4mPaired plot for features:\033[0m", '\n')
    sns.pairplot(data)
    plt.show()
```

---
## Dataset 1: Portfolio 💼 

![1) Portfolio_data_view](https://user-images.githubusercontent.com/35614192/213461305-176c6765-a54a-4e68-a0dd-8b268fc76023.png)

### Observations:
- Change the name of 'id' to 'offer_id' so it makes more sense going forward

- The `channels` column seems useful, but each value is a list. Therefore, we will apply OneHotEncoding (off=0, on=1) by leveraging scikit-learns `MultiLabelBinarizer` (this simplifies OneHotEncoding for a column containing lists in each row)

- Apply OneHotEncoding on `offer_type` column

- Change the `duration` from days to hours

Following is the code to accomplish the above steps:

```python
def clean_portfolio(portfolio):
    '''
    Function to clean the portfolio dataset:   
    
    Input: Original portfolio dataframe
    
    Output: Cleaned portfolio dataframe
    
    '''    

    # rename 'id' column to 'column_id' and set it as the first column
    portfolio.rename(columns={'id':'offer_id'}, inplace=True)
    first_column = portfolio.pop('offer_id')
    portfolio.insert(0, 'offer_id', first_column)
    
    # One-hot-encode for 'channels' column:
    # note: we could accomplish this by looping through a list ['web', 'email', 'social', 'mobile'] 
    # by expanding on the example below, but this would be suboptimal if the list was much longer
            # example: portfolio['web'] = portfolio['channels'].apply(lambda x: 1 if 'web' in x else 0)
    portfolio_copy = portfolio.copy()
    
    # One-hot-encode categorical variables (using sklearn.preprocessing.MultiLabelBinarizer)
    mlb = MultiLabelBinarizer()
    portfolio_copy = portfolio_copy.join(
        pd.DataFrame(mlb.fit_transform(portfolio_copy.pop('channels')),
                          columns=mlb.classes_,
                          index=portfolio_copy.index))
    
    # One-hot-encode 'offer_type' 
    portfolio_cleaned = pd.get_dummies(portfolio_copy, 
                                    columns = ['offer_type'])
    
    # change the duration from days to hours
    portfolio_cleaned['duration'] = portfolio_cleaned['duration'] * 24
    
    
    return portfolio_cleaned

portfolio = clean_portfolio(portfolio)
```

After cleaning portfolio, this is what it looks like:
![2) portfolio_cleaned](https://user-images.githubusercontent.com/35614192/213461465-d87a338b-8a62-47ff-9dd5-6ced033de392.png)

---
## Dataset 2: Profile 🗂
![3) profile intro](https://user-images.githubusercontent.com/35614192/213461518-398cf7f6-549b-47fd-ac5f-b5c44724c573.png)
![4) Profile plots](https://user-images.githubusercontent.com/35614192/213461584-d229ec0a-0c61-4cd1-a624-8c3f4667a6eb.png)

### Observations:

- `Age` seems to be approximately normally distributed apart from 2,175 outliers aged 118 years. I will save the `id's` of these customers because we may need them later when combining datasets:
```
cust_id_4_age_118 = profile[(profile['age']==118)][['id']].values
cust_id_4_age_118 = [item for sublist in cust_id_4_age_118 for item in sublist]
len(cust_id_4_age_118) # 2175
```

- Interestingly, both `gender` and `income` have the same % of missing values. -- After some sifting, I noticed that the 2,175 outliers ages 118 years were the ones missing `gender` and `income.`

- `became_member_on` should be in readable date time format

- Membership has been monotonically increasing every year from 2013-2018

- There is a decent overlap between Gender and Age for both males and females.

- Female income approximates closer to a normal distribution than that of males. Moreover, the female population has higher salaries overall.

- Loosely speaking, most customers are middle-aged

Following is the code to accomplish the above steps:

```python
def clean_profile(profile):
    '''
    Function to clean the profile dataset:

    Input: Original portfolio dataframe
    
    Output: Cleaned portfolio dataframe
    '''
    
    # drop rows where age == 118 & reset_index
    profile.drop(profile[profile.age == 118].index, inplace=True)
    profile.reset_index(drop=True, inplace = True)
    # drop rows with NaNs 
    profile.dropna(inplace=True)
    
    # rename 'id' column to 'customer_id' and set it as the first column
    profile.rename(columns={'id':'customer_id'}, inplace=True)
    first_column = profile.pop('customer_id')
    profile.insert(0, 'customer_id', first_column)
    
    # convert 'became_member_on' to more readable datetime format
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    
    # One-hot-encode 'gender'
    profile = pd.get_dummies(profile, columns = ['gender'])
    
    # Seperate 'age' into 'age_by_decade'
    profile['age_by_decade'] = pd.cut(profile['age'], bins=range(10,120,10),right=False, labels=['10s','20s', '30s', '40s', '50s','60s', '70s', '80s', '90s', '100s'])

    # OneHotEncoding feature 'age_by_decade'
    profile = pd.get_dummies(profile, columns=['age_by_decade'])
    
    # create income range & One-hot-encode it
    profile['income_by_range'] = pd.cut(profile['income'], bins=range(30000,140000,10000), right=False,\
                                        labels=['30ths','40ths', '50ths', '60ths', '70ths','80ths', '90ths',\
                                                '100ths', '110ths', '120ths'])
    profile = pd.get_dummies(profile,
                             columns=['income_by_range'], 
                             prefix='income', 
                             prefix_sep='_')

    # create a column for # days customer has been a member
    current_date = pd.to_datetime(datetime.today().strftime('%Y%m%d'))
    profile['membership_days'] = (current_date - profile['became_member_on']) / np.timedelta64(1,'D')
    
    # Add a 'Year' & 'Month' columns
    profile['became_member_year'] , profile['became_member_month'] = profile.became_member_on.dt.year.values, profile.became_member_on.dt.month.values  
    
    
    return profile


profile = clean_profile(profile)
```

After cleaning profile, this is what it looks like:

![5) Profile_cleaned](https://user-images.githubusercontent.com/35614192/213461659-b4457d71-2d89-4100-8405-2d0df5711dd8.png)


---

## Dataset 3: Transcript 📜
![6) Transcript_overview](https://user-images.githubusercontent.com/35614192/213461765-935dc153-b592-4448-af13-47e3d13110f8.png)


### Observations:
- Change the 'person' column name to 'customer_id.'

- Remove observations having customers with age 118

- Extract the 'offer_id' from the 'value' column

- Create separate columns for 'reward' and 'amount' (using the existing 'value' column)

- Drop 'value' after extracting data from it

- Group by customer id and reset the index
    - We want to distinguish people who KNOWINGLY participated in an offer as opposed to customers who participated in an offer by chance:
        - Successful offers <=>, where offer completed and offer viewed, are 1≤.


Following is the code to accomplish the above steps:
```python
def clean_transcript(transcript):
    '''
    Function to clean the transcript dataset:

    Input: Original transcript dataframe
    
    Output: Cleaned transcript dataframe
    '''

    # change 'person' column name to 'customer_id'
    transcript.rename(columns={'person':'customer_id'},inplace=True)    
    
    # Remove observations having customers with age 118
    transcript = transcript[~transcript['customer_id'].isin(cust_id_4_age_118)]
    
    # Extract the 'offer_id' from 'value'
    transcript['offer_id'] = transcript['value'].apply(lambda x: x['offer_id'] if 'offer_id' in x else 
                                                       (x['offer id'] if 'offer id' in x else None))
    
    # Create seperate columns for 'reward' and 'amount' (using existing 'value' column)
    for i in ['reward','amount']:
        transcript[i] = transcript['value'].apply(lambda x: x[i] if i in x else None)

    # drop 'value' after extracting data from it
    transcript.drop('value', axis=1, inplace=True)
    
    # Group by customer id and reset index
    df = transcript.groupby(['customer_id','offer_id','event'])['time'].count().unstack()
    df.reset_index(level=[0,1],inplace = True)

    # This is important because we want to distinguish customers who KNOWINGLY participated in an offer as opposed to customers who participated by chance 
    df['successful_offer'] = df['offer viewed'] * df['offer completed']
    df['successful_offer'] = df['successful_offer'].apply(lambda x: 1.0 if x > 0 else 0.0)
    df.drop(['offer completed'],axis=1, inplace = True)

    # Fill all NA with '0'
    df.fillna(0.0, inplace=True)
    
    return df

transcript = clean_transcript(transcript)
transcript
```
![7) Transcript_cleaned](https://user-images.githubusercontent.com/35614192/213461874-a7112531-9d70-4569-9086-9a25f3112e29.png)

---
# Merge data sets and visualize:

```python
# Merge all dataframes together:

# merge transcript with portfolio on 'offer_id'
master_df = transcript.merge(portfolio, how='left', on='offer_id')
# merge master with profile on 'customer_id'
master_df = master_df.merge(profile, how='left', on='customer_id')

# Clean up 'offer_id' column
ten_unique_ids = list(master_df['offer_id'].unique())
for i in range(len(ten_unique_ids)):
    master_df['offer_id'] = master_df['offer_id'].apply(lambda x: f'{i+1}' if x == ten_unique_ids[i] else x)
    
master_df.head()
```

![8) master head](https://user-images.githubusercontent.com/35614192/213461927-f5a223d4-514c-4b36-9831-f327fe1fe58c.png)
![9) master viz](https://user-images.githubusercontent.com/35614192/213461948-314f1a12-60b3-449f-b80f-91a43ca41142.png)

---
# 🤖 Data Modeling
First, we must determine our predictor features (X) and target variable (y). Knwoing that ML algorithms only perform with numeric data, I will remove any string or time features:

```python
# Drop meaningless features
master_df.drop(['customer_id','email','became_member_month','became_member_year','became_member_on'], axis=1, inplace=True)

# predictive variables
X = master_df.drop(columns=['successful_offer'])
# target variable
y = master_df['successful_offer']

# Correlation
plt.figure(figsize=(10,10))
sns.heatmap(X.corr(),square=True, cmap='viridis');
plt.title('Correlation Heatmap', fontsize=20);
```

![10) Corr viz](https://user-images.githubusercontent.com/35614192/213461993-14558482-15f7-4db0-849f-e0e2dba7d2b6.png)

```python
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

features_to_scale = ['membership_days', 'income', 'reward', 'difficulty', 'duration']

def scale_features(df, features_to_scale):
    '''
    Replace "unscaled" features in a df with scaled features
    
    INPUT:
    * df (dataframe): df containing features to scale
    * features_to_scale (list): list of features to scale
    
    OUTPUT:
    * df_scaled (dataframe): updated df containing scaled features (original "unscaled" features replaced)
    '''
    
    # df with features to scale
    df_features_scale = df[features_to_scale]
    
    # Initialize scaler and apply feature scaling to df
    scaler = MinMaxScaler()
    df_features_scale = pd.DataFrame(scaler.fit_transform(df_features_scale), columns=df_features_scale.columns, index=df_features_scale.index)
    
    # Drop orignal features from df and replace with scaled features 
    df = df.drop(columns=features_to_scale, axis=1)
    df_scaled = pd.concat([df, df_features_scale], axis=1)
    
    return df_scaled

# Applying the function to features:
X_train_scaled = scale_features(X_train, features_to_scale)

# View distribution of target class:
print(round(y_train.value_counts(normalize=True)*100,2))
```

![11) value_counts%](https://user-images.githubusercontent.com/35614192/213462033-f3864551-c852-4885-b4db-3874a5222716.png)

- Target variable is distributed at 53.36% and 46.64% for classes 1 and 0, respectively. Therefore, our data is approximately balanced, and we do not have to implement techniques to deal with unbalanced data (i.e., oversampling minority classes, etc.)

- I will use the F1-Score to measure the model's competency because the F1-Score metric is "the harmonic mean of the precision and recall metrics." 

Next, we train classifiers and determine the best estimator using GridSearchCV and times each selected model:

```python
from datetime import datetime as dt

def fit_classifier(clf, param_grid, X=X_train_scaled.values, y=y_train.values):
    '''
    1) Fit a classifier on training data using GridSearchCV and provide F1-Score
    2) Collect timed results for each classifier
    
    INPUT:
    * clf               : classifier to fit
    * param_grid (dict) : parameters to use with GridSearchCV
    * X                 : X train values
    * y                 : y train target values
    
    OUTPUT:
    * classifier results
    '''
    
    start_time = dt.now()
    
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1', cv=5, verbose=0)
    print("Fitting {} :".format(clf.__class__.__name__))
    grid.fit(X, y)
    total_time = (dt.now()-start_time).seconds
    
    print(clf.__class__.__name__)
    print('Duration (seconds): {}'.format(total_time))
    print('Best F1_Score: {}'.format(round(grid.best_score_, 3)))
    print("-"*50)
    
    return total_time, grid.best_score_, grid.best_estimator_
```

#  🕵️‍♂️ Model Evaluation

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# initialize classification models:
rfc = RandomForestClassifier(random_state=123) # RandomForestClassifier
gbc = GradientBoostingClassifier(random_state=123) # GradientBoostingClassifier
ada = AdaBoostClassifier(random_state=123) # AdaBoostClassifier

cl_names = []
cl_scores = []
cl_best_ests = []
cl_time_taken = []
cl_dict = {}

for classifier in [rfc, ada, gbc]:
    total_time, best_score, best_est = fit_classifier(classifier, {})
    cl_names.append(classifier.__class__.__name__)
    cl_scores.append(best_score)
    cl_best_ests.append(best_est)
    cl_time_taken.append(total_time)

```
![12) model results](https://user-images.githubusercontent.com/35614192/213462098-4724ca60-83af-475d-b3cb-036eee486c1a.png)


The GradientBoostingClassifier performs marginally better than the other models. Although it took slightly longer to train, 33 seconds is a significant tradeoff, especially when knowing specific models can take hours or even days.

`Caveat: I did not fully follow through with parameter tuning due to time constraints. The cost vs. benefit was not favorable.`
 
After proceeding with general model implementation, below are the visualized results of the model:

![13) final model viz](https://user-images.githubusercontent.com/35614192/213462362-59631349-08e9-4f57-8c87-4b5a058c7426.png)


#  Model Conclusion:
- 'Offer viewed' is deemed the most important feature by the model. Thus, communicating this to the marketing division to `increase brand awareness` via offers may be a reasonable goal.

- See where informational offers can be improved because that seems to be working

- The Receiver Operator Characteristic (ROC) provides a simple way to summarize all of the information of the confusion matrix. Overall, it illustrates that the proportion of correctly classified samples (true positives) is greater than the proportion of the samples that were incorrectly classified (false positives). Performance evaluation would depend on what values the team would like to focus on: True Positive Rate (TPR) or False Positive Rate (FPR). Then we can find out which threshold values [0, 0.2, 0.4, 0.6, 0.8, 1.0] we should put emphasis on. I think an FPR of 0.2 may be decent in this case. I suspect the model may be leaning towards predicting the survived class of 1 as opposed to 0, and the Area under the ROC curve is 0.932. Although it is fairly close to 1, I think the model can be tuned to classify the 0 class better.

- The classification report indicates that precision, recall, and f1-score are all approximately 80≤, so that is a relatively decent starting point 

# Further Improvements:
- There is room for more feature engineering within the data. Doing so may likely make the models more robust
- To find optimal customer demographics, it would be nice to have a few more features of a customer. Thus, experimenting with more data may prove helpful. 

That's all for now folks, until next time.
