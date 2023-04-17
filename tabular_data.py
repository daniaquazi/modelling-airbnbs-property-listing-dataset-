import pandas as pd

class hello():

    def remove_rows_with_missing_ratings(listings):
        # Reading the CSV file and assign it to the 'listings' variable
        listings = pd.read_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/AirBnbData.csv")
        # Dropping rows with missing values
        listings = listings.dropna(subset=['Cleanliness_rate', 'Accuracy_rate', 'Communication_rate', 'Location_rate', 'Check-in_rate', 'Value_rate'])
        return listings

    def combine_description_strings(self):
        # Call function to remove rows with missing ratings
        listings = self.remove_rows_with_missing_ratings()
        # Drop rows with missing values
        listings = listings.dropna(subset=['Description'])
        # Removing unnecessary text from the 'Description' column using lambda functions
        listings['Description'] = listings['Description'].apply(lambda x: x[1:-1])
        listings['Description'] = listings['Description'].apply(lambda x: x.replace('About this space', ''))
        listings['Description'] = listings['Description'].apply(lambda x: x.replace("''", ''))
        listings['Description'] = listings['Description'].apply(lambda x: x.replace(",", ''))
        return listings

    def set_default_feature_values(self):
        # Call function to combine list items into the same string
        listings = self.combine_description_strings()
        # Filling missing values in specified columns with the value 1
        listings['guests'] = listings['guests'].fillna(1)
        listings['beds'] = listings['beds'].fillna(1)
        listings['bathrooms'] = listings['bathrooms'].fillna(1)
        listings['bedrooms'] = listings['bedrooms'].fillna(1)
        return listings

    def clean_tabular_data(self, listings):
        # Call the set_default_feature_values function
        listings = self.set_default_feature_values()
        # Drop unecessary columns
        listings = listings.drop(['Category', 'ID','Title', 'Description','Amenities','Location','url'],axis=1)
        # listings = listings.drop(['ID'],axis=1)
        return listings

    def load_airbnb(self, df, label):
        # Dropping the label from the features
        listings = df.drop(label, axis = 1)
        # listings = listings.drop(['Category', 'ID','Title', 'Description','Amenities','Location','url'],axis=1)
        return listings, df[label]

if __name__ == "__main__":
    s = hello()
    # Reading the AirBnb file
    df = pd.read_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/AirBnbData.csv")
    # Cleaning the data using clean_tabular_data function
    clean_data = s.clean_tabular_data(df)
    print(clean_data)
    # Saving the cleaned data to a csv file
    clean_data.to_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/clean_tabular_data.csv")
