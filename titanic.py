from tensorflow import keras
from keras.layers import Dense, Input  
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import os

IMPORT_PATH = "data"
EXPORT_PATH = "submissions/nn"
TRAIN_NAME = "train.csv"
TEST_NAME = "test.csv"

class NNModel():
    def __init__(self, num_inputs: int,  hidden_layers=[], activations=[], name="titanic-model") -> None:
        self.hidden_layers = hidden_layers
        self.name = name

        num_layers = len(self.hidden_layers)

        keras.backend.clear_session()
        inputs = Input(shape=(num_inputs,))
        if num_layers > 0:
            dense = Dense(self.hidden_layers[0], activation=activations[0])
            x = dense(inputs)
            for i in range(1, num_layers):
                x = Dense(hidden_layers[i], activation=activations[i])(x)
        else: 
            x = inputs
        outputs = Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        self.model = model

    def compile(self, loss, optimizer, metrics=["accuracy"]):
        self.model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = metrics,
        )

    def summary(self):
        self.model.summary()

    def train(self, X,  y, batch_size=20, epochs=2, validation_split=0.2):
        return self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, X):
        predictions = self.model.predict(X) 
        return np.round(predictions.flatten()).astype(np.int8)
class DataLoader():

    def __init__(self, import_path: str, export_path: str) -> None:
        self.import_path = import_path
        self.export_path = export_path

    def import_data(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.import_path, file_name)
        return pd.read_csv(file_path)
        
    def export_data(self, predictions: pd.DataFrame, file_name: str) -> None:
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)
        predictions.to_csv(f"{self.export_path}/{file_name}", index=False)


class Preporcessor():

    def substrings_in_string(self, string: str):
        title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer'] 
        for substring in title_list:
            if string.find(substring) != -1:
                return substring
        return np.nan
    
    def simplify_title(self, df: pd.DataFrame):
        title = df["Title"]
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if df['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    def extract_deck(self, cabin):
        if isinstance(cabin, float):
            return "Unknown"
        return cabin[0]


    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add title feature.
        return_df = df.copy()
        return_df["Title"] = return_df["Name"].map(lambda x: self.substrings_in_string(x))
        return_df['Title'] = return_df.apply(self.simplify_title, axis=1)
        print("Added title feature...")
        
        # Remove unnecessary columns.
        return_df.drop(["Name"], axis=1, inplace=True)
        return_df.drop(["PassengerId"], axis=1, inplace=True)
        print("Removed unnecessary columns...")
        
        # Create feature deck.
        return_df["Deck"] = return_df["Cabin"].map(lambda x: self.extract_deck(x))
        return_df.drop(["Cabin"], axis=1, inplace=True)
        print("Created deck feature...")
        
        # Fix age feature.
        median_ages_by_deck_sex = return_df.groupby(["Deck", "Sex"]).median()["Age"]
        return_df["Age"] = return_df.apply(lambda x: median_ages_by_deck_sex[x["Deck"]][x["Sex"]] if pd.isnull(x["Age"]) else x["Age"], axis=1)
        print("Fixed possible na values for age...")
            
        # Fix fare feature.
        median_fares_by_deck = return_df.groupby("Deck").median()["Fare"]
        return_df["Fare"] = return_df.apply(lambda x: median_fares_by_deck[x["Deck"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1)
        print("Fixed possible na values for fare...")
        
        # Drop Ticket.
        return_df = return_df.drop(["Ticket"], axis=1)
        print("Removed ticket column...")
            
        # Categorical features to dummies.
        return_df = pd.get_dummies(return_df)
        print("Created dummies for categorical features...")
        
        # Return data
        X, y = ( return_df.drop("Survived", axis=1), return_df["Survived"] ) if "Survived" in return_df else ( return_df, None )
        print("Transformation done!")
        
        return X, y

    def adjust_missing_cols(self, train: pd.DataFrame, test: pd.DataFrame):
        missing_cols = set(train.columns) - set(test.columns)
        for c in missing_cols:
            test[c] = 0.0
        test = test[train.columns]
        return test
    
    def scale_data(self, df, scaler):
        return scaler.fit_transform(df)

    def create_submission_df(self, predictions):
        starting_passenger_id = 892
        last_passenger_id = 1309
        passenger_ids = np.arange(starting_passenger_id, last_passenger_id + 1)

        return pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions
        })
        






def main():
    data_loader = DataLoader(IMPORT_PATH, EXPORT_PATH)
    preproecessor = Preporcessor()

    df_train = data_loader.import_data(TRAIN_NAME)
    df_test = data_loader.import_data(TEST_NAME)

    X_train, y_train = preproecessor.clean(df_train)
    X_test, y_test = preproecessor.clean(df_test)

    X_test = preproecessor.adjust_missing_cols(X_train, X_test)


    scaler = MinMaxScaler()
    X_train = preproecessor.scale_data(X_train, scaler)
    X_test = preproecessor.scale_data(X_test, scaler)

    keras.backend.clear_session()

    test_layers = [
        [64],
        [32],
        [16],
        [8],
        [64, 32],
        [64, 16],
        [64, 8],
        [32, 16],
        [32, 8],
        [16, 8],
        [64, 32, 16],
        [64, 32, 8],
        [64, 16, 8],
        [64, 32, 16, 8]
    ]

    for test_layer in test_layers:
        model = NNModel(X_train.shape[1], test_layer, ["relu" for i in range(len(test_layer))])
        print("######### MODEL WITH LAYERS " + str(test_layer) + " #########")
        model.compile(keras.losses.BinaryCrossentropy(), keras.optimizers.Adam())
        model.summary()
        history = model.train(X_train, y_train)

        predictions = model.predict(X_test)
        submissions = preproecessor.create_submission_df(predictions)
        out_file_name = f"nn_submissions_{'_'.join( [ str(layer) for layer in test_layer ])}.csv"
        data_loader.export_data(submissions, out_file_name)


    


main()