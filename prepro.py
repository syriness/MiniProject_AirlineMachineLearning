import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(airline, airline2):
    encoder = LabelEncoder() 
    encoder.fit(airline["satisfaction"])
    airline["satisfaction"] = encoder.transform(airline["satisfaction"])
    
    airline_test = airline2["satisfaction"]
    airline_test_X = airline2.iloc[:, 8:24]

    encoder2 = LabelEncoder()
    encoder2.fit(airline_test)
    airline_test = encoder2.transform(airline_test)

    airline_test_X.info()
    airline_test_X.astype(int)
    
    airline_score = airline.iloc[:, 8:24]
    
    airline_score.info()
    airline_score.astype(int)
    airline_score["satisfaction"] = airline["satisfaction"]
    
    X = airline_score.drop(["satisfaction"], axis=1)
    y = airline_score["satisfaction"]
    
    return X, y, airline_test_X, airline_test
