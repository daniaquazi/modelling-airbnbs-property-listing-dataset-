import pandas as pd

class hello():

    def remove_rows_with_missing_ratings(listings):
        listings = pd.read_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/AirBnbData.csv")
        listings = listings.dropna(subset=['Cleanliness_rate', 'Accuracy_rate', 'Communication_rate', 'Location_rate', 'Check-in_rate', 'Value_rate'])
        return listings

    def combine_description_strings(self):
        listings = self.remove_rows_with_missing_ratings()
        listings = listings.dropna(subset=['Description'])
        listings['Description'] = listings['Description'].apply(lambda x: x[1:-1])
        listings['Description'] = listings['Description'].apply(lambda x: x.replace('About this space', ''))
        listings['Description'] = listings['Description'].apply(lambda x: x.replace("''", ''))
        listings['Description'] = listings['Description'].apply(lambda x: x.replace(",", ''))
        return listings

    def set_default_feature_values(self):
        listings = self.combine_description_strings()
        listings['guests'] = listings['guests'].fillna(1)
        listings['beds'] = listings['beds'].fillna(1)
        listings['bathrooms'] = listings['bathrooms'].fillna(1)
        listings['bedrooms'] = listings['bedrooms'].fillna(1)
        return listings

    def clean_tabular_data(self, listings):
        listings = self.set_default_feature_values()
        return listings

    def load_airbnb(self, df, label):
        listings = df.drop(label, axis = 1)
        listings = listings.drop(['ID','Category','Title','Description','Amenities','Location','url'],axis=1)
        return listings, df[label]

if __name__ == "__main__":
    s = hello()
    df = pd.read_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/AirBnbData.csv")
    clean_data = s.clean_tabular_data(df)
    print(clean_data.loc[0])
    clean_data.to_csv("clean_tabular_data.csv")
