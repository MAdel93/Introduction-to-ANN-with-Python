# Preprocessing

In this project, we work on the function of multilayer perceptron neural networks; 
The attached data set includes the data obtained from the environmental sensors of smart homes. 
The label of this data set is binary and shows whether the fire alarm has been activated based on the data received from the built-in sensors. 
The purpose of working with this data set is to train a model based on the multilayer perceptron neural network to predict the alarm's activation.

## Estimate Missing Value, Normalization, Remove Outliers 
### Missing Value
In the first step of this project, we have to find missing values, shown by "Nan", and then estimate by one of the estimation techniques.
There are many methods to estimate the missing values. You can use the mean and median or a random value to substitute the missing value, 
but these methods have good accuracy. They have not. Regression, KNN, or PCA methods can also be used to estimate missing values, which are highly accurate, 
but face complications.

In this project, we use random imputation.

![image](https://user-images.githubusercontent.com/118474020/202857350-45759611-66fe-446d-9234-8056680d96fe.png)
![image](https://user-images.githubusercontent.com/118474020/202857392-f82345a7-123d-457b-9f81-f3683d1780cd.png)
### Normalization
Normalization, which refers to processes that achieve scales between zero and one, is one of the most critical steps in preprocessing. 
This cause eliminates redundant data, minimizing data modification errors and simplifying the query process.
### discretization
Then we have to convert all the object-type data into numerical data (discretization). 
For this reason, we should replace discrete values in alphabetical order with numbers.
![image](https://user-images.githubusercontent.com/118474020/202857756-b1ece459-27dd-4db1-9ae7-4d45bd4d2b40.png)
### Outliers
Outlier data refers to data that has an abnormal distance from other data or, in general, the designed classes in the data set. In other words, outliers are values that are prominently and distinctly placed in the main pattern of the data set or graph of data.

Therefore, to deal with the negative effects of outlier data, which has been partially improved in the normalization phase, various methods have been used, some of which are as follows:

1- Standard Deviation: In this method, the distance of the data from the average has been examined. So, the data far from the average of more than a certain amount is known as outlier data and is removed.

2- Z-score method: This is based on the ratio of the distance between data and the average to the standard deviation.

3- Interquartile range (IQR): It is one of the measures methods of distribution which is equal to the distance between the first and third quartiles. IQR is an index for distribution that calculates the difference between the 25th and 75th percentiles.

4- Percentile: It is a number where a certain percentage of scores fall below that number.

In this part, method number 3 is used to find outlier data along with the Kmean algorithm. For this reason, we first categorize the data using the Kamen algorithm as follows.

Now, in the KMean, the number of categories after several repetitions is set to 10. Then the distance of each data with the centre of its cluster is calculated. 

Finally, outliers are identified and removed using the interquartile range method in the obtained intervals.

At the final stage in preprocessing we have to split dataset into three part for training, test and evaluation phases.

## Training

In this phase, we should design an MLP neural network and then train it with the prepared dataset from the previous step. The designed MLP neural network consists of three hidden layers, as shown below, and its precision reaches the peak of %82.

![image](https://user-images.githubusercontent.com/118474020/202861670-e239ed6d-5a14-47ab-a9e5-618cc46cebb7.png)

