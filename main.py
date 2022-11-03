import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#from sklearn.linear_model import LinearRegression
#from sklearn import train_test_split


def predict_attendance(filename, RefineData=False):
    # One hot encoding of categoprial data - here "Day of The week"
    #daysofweek = pd.DataFrame({'day': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']})
    #daysofweek_encoded= pd.get_dummies(daysofweek)
    #print (daysofweek_encoded)
    #idx     day_Friday  day_Monday  day_Saturday  day_Sunday  day_Thursday  day_Tuesday  day_Wednesday
    #0           0           0             0           1             0            0              0
    #1           0           1             0           0             0            0              0
    #2           0           0             0           0             0            1              0
    #3           0           0             0           0             0            0              1
    #4           0           0             0           0             1            0              0
    #5           1           0             0           0             0            0              0
    #6           0           0             1           0             0            0              0

    dataset = pd.read_csv(filename)

    if bool(RefineData) == True:
        dataset = dataset[dataset.Weekend !=1]
        dataset = dataset[dataset.Holiday != 1]
        #dataset = dataset[dataset.Celebrations !=1]
        #dataset = dataset[dataset.Workday != 0]    

    # drop the few irrevant columns 
    dataset = dataset.drop(columns=["DATE" , "UTC_TMAX", "UTC_TMIN"]) #,"Weekend","Holiday", "Workday", "Celebrations"])
    #print (dataset[1])

    target_name = "Count"
    target = dataset[target_name]

    data = dataset.drop(columns=[target_name])

    # ## Selection based on data types
    #
    # We will separate categorical and numerical variables using their data
    # types to identify them,

    from sklearn.compose import make_column_selector as selector

    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = categorical_columns_selector(data)

    # We need to treat data differently depending on their nature (i.e. numerical or categorical).
    #
    # Use Scikit-learn `ColumnTransformer` class which will send specific columns to a specific transformer, 
    # making it easy to fit a single predictive model on a dataset that combines both kinds of variables together
    # (heterogeneously typed tabular data).
    #
    # We first define the columns depending on their data type:
    #
    # * **one-hot encoding** will be applied to categorical columns. Besides, 
    # we use `handle_unknown="ignore"` to solve the potential issues due to rare categories.
    # * **numerical scaling** numerical features which will be standardized.
    #
    # First, let's create the preprocessors for the numerical and categorical
    # parts.

    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    # Now, we create the transformer and associate each of these preprocessors
    # with their respective columns.

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)])


    # A `ColumnTransformer` does the following:
    #
    # * It **splits the columns** of the original dataset based on the column names
    #   or indices provided. We will obtain as many subsets as the number of
    #   transformers passed into the `ColumnTransformer`.
    # * It **transforms each subsets**. A specific transformer is applied to
    #   each subset: it will internally call `fit_transform` or `transform`. The
    #   output of this step is a set of transformed datasets.
    # * It then **concatenates the transformed datasets** into a single dataset.

    # The important thing is that `ColumnTransformer` is like any other
    # scikit-learn transformer. In particular it can be combined with a classifier
    # in a `Pipeline`:

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(preprocessor, LinearRegression())
    model
    #print (model.get_feature_names_out)
    #model.get_params
    #print(model._get_params)

    # - the `fit` method is called to preprocess the data and then train the
    #   classifier of the preprocessed data;
    # - the `predict` method makes predictions on new data;
    # - the `score` method is used to predict on the test data and compare the
    #   predictions to the expected test labels to compute the accuracy.
    #
    # Let's start by splitting our data into train and test sets.

    from sklearn.model_selection import train_test_split

    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=42)

    #print (data_test)

    #
    # Now, we can train the model on the train set.

    _ = model.fit(data_train, target_train)


    # Then, we can send the raw dataset straight to the pipeline. Indeed, we do not
    # need to make any manual preprocessing (calling the `transform` or
    # `fit_transform` methods) as it will be handled when calling the `predict`
    # method. As an example, we predict on the five first samples from the test
    # set.


    #data_test.head()


    target_predict = model.predict(data_test)#[:5]
    #print(target_predict)
    #print (target_test)


    #target_test[:5]


    # To get directly the accuracy score, we need to call the `score` method. Let's
    # compute the accuracy score on the entire test set.

    # calculating accuracy score
    #from sklearn.metrics import accuracy_score
    #print (target_predict, target_test.to_numpy(dtype=np.float64))
    #accuracy_score = accuracy_score(target_predict > 0.5,target_test.to_numpy())
    #print('accuracy score : ',accuracy_score)

    accuracy_score = model.score(data_test, target_test)
    print ("Accuracy Score : "f"{accuracy_score*100:.2f}%")

    #plt.scatter(target_predict, target_test, color ='b')
    #plt.plot(data_test.tolist()[4], target_predict, color ='k')
    
    #plt.show()

    #Regression plot of our model.
    # A regression plot is useful to understand the linear relationship between two parameters. 
    # It creates a regression line in-between those parameters and then plots a scatter plot of those data points.

    sns.regplot(x=target_test,y=target_predict,ci=None,color ='red')
    plt.show()

    #We can also visualize comparison result as a bar graph using the below script :
    #Note: As the number of records is not large, for representation purpose I’m taking all records, otherwise 
    # I also could show just 25 records
    df = pd.DataFrame({'Actual': target_test, 'Predicted': target_predict})
    df
    #df1 = df.head(25)
    df1=df
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    #plt.scatter(data_test, target_test,  color='gray')
    #plt.plot(data_test, target_predict, color='red', linewidth=2)
    #plt.show()

    # ## Evaluation of the model with cross-validation
    #
    # A predictive model should be evaluated by cross-validation. 
    # Our model is usable with the cross-validation tools of scikit-learn as any other predictors:

    from sklearn.model_selection import cross_validate

    #cv_results = cross_validate(model, data, target) #, cv=5)
    #cv_results

    #scores = cv_results["test_score"]
    #print("The mean cross-validation accuracy is: "
    #    f"{scores.mean():.3f} ± {scores.std():.3f}")

    # The compound model has a higher predictive accuracy than the two models that
    # used numerical and categorical variables in isolation.

    #The final step is to evaluate the performance of the algorithm. 
    # This step is particularly important to compare how well different algorithms perform on a particular dataset. 
    # For regression algorithms, three evaluation metrics are commonly used

    mean_test = target_test.mean() #pd.DataFrame({'Actual': target_test}).mean()
    print("Mean value of Attendance count: ",round(mean_test))
    err_abs=metrics.mean_absolute_error(target_test, target_predict)
    print('Mean Absolute Error (Attendance count):', round(err_abs), "i.e. "f"{err_abs*100/mean_test:.2f}","% of Mean")
    ##print('Mean Squared Error (Attendance count):', round(metrics.mean_squared_error(target_test, target_predict)))
    err_rms= np.sqrt(metrics.mean_squared_error(target_test, target_predict)) 
    print('Root Mean Squared Error (Attendance count):', round(err_rms))

    #Manual Testing
    # Input data in format ['TMAX_C', 'TMIN_C', 'Humidity_pc', 'RAINFALL_mm', 'DAY', 'Workday', 'Weekend', 'Holiday', 'Celebrations']
    input_test = np.array([[34.6,24.8,64,0,"Monday",1,0,0,0]])
    #print(list(data_train.columns))
    input_df = pd.DataFrame(input_test, columns =list(data_train.columns))
    #print(input_df)
    print("Predicted Attendance: ", model.predict(input_df))

#Invoke preduct function
# To Pass second argument as True if we want to refine input by removing data with noise 
predict_attendance("dataset_mixedtype.csv", False)

