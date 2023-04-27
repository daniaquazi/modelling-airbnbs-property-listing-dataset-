# modelling-airbnbs-property-listing-dataset-

Milestone 3

This milestone is done in the tabular_data.py file.

The first task was to clean the dataset that was provided. There are quite a lot of issues with the dataset such as many missing values. So, the first task was to create a I function called  “remove_rows_with_missing_ratings” which reads in a CSV file of Airbnb data and drops any rows with missing values.

<img width="381" alt="image" src="https://user-images.githubusercontent.com/46778501/234930893-05dc3868-fc81-44ef-b29f-df3c17c687e9.png">




One column in the dataset contains descriptions however this is in the wrong format (in the form of values which Pandas doesn't recognize). When reading data from a CSV file, Pandas reads in the data as strings anyways, and any data that looks like a list is simply read in as a string whose contents are valid Python lists. So, because of this, the next task was to create a function called, combine_description_strings in order to combine all the strings in the "Description" column into a single string, removing unnecessary text, and removing commas.






The next task was to create a function called, set_default_feature_values which fills in missing values in the "guests", "beds", "bathrooms", and "bedrooms" columns with the value 1. This was done because it ensures that all rows have a value in those columns, which makes it easier to analyse the data and avoid errors that might occur from having missing or null values. It can also indicate that there is at least one of each feature in the listing.






The final task was to create a function called, clean_tabular_data which takes the dataset as input and calls the "set_default_feature_values()" method to clean the dataset. It then drops unneeded columns ("Category", "ID", "Title", "Description", "Amenities", "Location", and "url") and returns the cleaned dataset. By doing this, it makes it easier to reuse the code, instead of having to call each function one by one.
 
Milestone 4

This milestone is done in a file called modelling.ipynb.

In this milestone, I created machine learning models to predict the price per night and to measure the RMSE and R2 score.

First, I imported the AirBnB data, cleaned this and set the target variable to “Price_Night”. I then, split the data into training and testing sets, trained an SGD regressor model on the training data, made predictions on the testing set, and finally made predictions on the training set.



I then calculated the mean squared error (which came out to be 6355620388.999379 for the training set and 6235928567.913321 for the test set) 
Which means that the predicted values are not close at all to the actual values.

After calculating the r2 score (which came out to be -3409520697768573.5 for the training set and -1267386333541525.0 for the test set), I came to the conclusion that the model performed very badly because a good R-squared value should be close to 1.

My next task was to create a function that tuned hyperparameters for a regression model. The function iterates through all possible combinations of hyperparameters, trains the model using the training data, and evaluates its performance on the test data using mean squared error (MSE). 







The best hyperparameters are the ones that give the lowest MSE. My function then returned the best model, hyperparameters, and the MSE.





The next task was to save the model, hyperparameters and metrics to individual files.






I tried improving the performance of the model by using different models provided by sklearn: decision trees, random forests, and gradient boosting. After running each of these models, I came to the conclusion that random forest regressor had the best performance by comparing the metrics.

I created a long function, find_best_model which evaluated the best model and it returned the loaded models, a dictionaries of the hyperparameters, and dictionaries of the performance metrics.

Milestone 5

I did exactly the same thing as milestone 4 but here however, I also computed the key performance measures (including F1, precision, recall, and the accuracy score for training and test sets and trained the classification models instead of the regressor versions of decision trees, random forests, and gradient boosting.

Milestone 6

This milestone is done in a file called pytorch.py.

In this milestone, overall, I created a PyTorch model to predict Airbnb nightly prices based on different features.

I created a custom PyTorch dataset that is designed to load and preprocess image and price data from Airbnb listings. I have three different methods in this class:

-	__init__ method: the dataset is initialized by loading the airBnB csv data, removing columns that aren’t needed and storing the 'Price_Night' column as the target variable.
-	__len__ method: this defines the length of the dataset.
-	__getitem__ method: this gets the features and the label and returns a tuple of them as tensors.

The next thing I did was to create a dataloader. For this, first I needed to create an instance of the AirbnbNightlyPriceImageDataset class, then split the dataset into training, testing, and validation subsets and then finally create the dataloaders for the training, testing, and validation subsets.

The next thing I did was to create a LinearRegression class which implements a linear regression model in PyTorch. By inheriting from torch.nn.Module, the LinearRegression class has access to different functionalities and can create its own architecture for a linear regression model. In this class, the architecture consists of a single linear layer that maps input data to output predictions. The class also includes methods for performing forward and backward passes.

I defined a training loop which took the model and the hyperparameters as input to basically train the model. It trained the model for the specified number of epochs using the training data, optimiser and loss function, and returned the trained model. During each epoch, it iterated over the training data in batches, calculated the loss for each batch, performed backpropagation to update the model's parameters, and logged the average loss and accuracy for the epoch. At the end of each epoch, it evaluated the model on the validation data and logged the validation loss and accuracy. Once the training completed, it returned the trained model. Finally, I created a dictionary to contain metrics which I returned.

I created a YAML file which contained different configurations (including optimiser, learning rate, hidden layer width and depth. This was then read and turned into a dictionary. It was used by a function called generate_nn_configs that used the hperparameters to generate different combinations.

I then created a function called find_best_nn which sequentially trained the model using different combinations of the hyperparameters. It then checked if the R-squared value of the current trained model was better than the current best R-squared value and if so, it updated the best_r2, best_model, best_hyperparams, and best_metrics variables with the values from the current trained model. Finally, I saved the model, hyperparameters and metrics in their own files.

Milestone 7

This milestone is done in a file called pytorch_different_dataset.py.

For this milestone, I changed the target variable from “Price_Night” to a different variable.
