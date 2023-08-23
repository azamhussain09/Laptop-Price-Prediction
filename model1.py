import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib



df = pd.read_csv('Preprocessed.csv')
X = df.drop(columns=['Price'])
y = np.log(df['Price'])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

#print('R2 score',r2_score(y_test,y_pred))
#print('MAE',mean_absolute_error(y_test,y_pred))


# Assuming you have already trained your 'pipe' pipeline

# Save the trained pipeline
joblib.dump(pipe, 'C:/Users/AZAM HUSSAIN/OneDrive/Desktop/Placements and projects/ML Projects/trained_pipeline.pkl')
