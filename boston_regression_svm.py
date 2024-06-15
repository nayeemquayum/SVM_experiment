import pandas as pd
import numpy as np
import plotly_express as px
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

boston_df = pd.read_csv('data/BostonHousing.csv')
print(boston_df.info())
boston_df = boston_df.dropna()
print("After removing Nan, the dataframe:",boston_df.info())
#drop all features except rm, lstat and dis
boston_3D_df=boston_df.drop(['crim','zn','indus','chas','nox','age','rad','tax','ptratio','b'], axis='columns')
#draw data 3D space
# fig= px.scatter_3d(data_frame=boston_3D_df,\
#                    x='rm',
#                    y='lstat',
#                    z='dis',
#                    color='medv')
# fig.update_layout(margin=dict(l=20,r=20,b=20))
# fig.show()
#GridSearchCV
X = boston_3D_df.iloc[:,0:3]
y = boston_3D_df.iloc[:,3]
svr_model=SVR()
#find the optimal hyperparameters values for SVR using GridSearchCV
svr_parameters = {'kernel':('linear', 'rbf','poly','sigmoid'),
                  'C':[0.1,.05,1,3,5,7,9,10]
                  }
svr_grid=GridSearchCV(svr_model,svr_parameters,verbose=4,refit=True)
svr_grid.fit(X,y)
print("GridSearchCV best parameters:",svr_grid.best_params_)
print("GridSearchCV best parameters score:",svr_grid.best_score_)

# mean_corss_validation_score= cross_val_score(svr_model, X, y, cv=10,scoring='r2').mean()
# print("Mean corss validation score for R2 of SVR:",mean_corss_validation_score)
svr_model_2=SVR(kernel='rbf',C=9)
#split data in train and test
X_train, X_test, y_train, y_test = train_test_split(boston_3D_df.drop(columns=['medv']),
                                                    boston_3D_df["medv"], test_size=0.2,random_state=20)
svr_model_2.fit(X_train,y_train)
print("SVR model score (with rbc kernel and C=9):",svr_model_2.score(X_train, y_train))
y_prediction=svr_model_2.predict(X_test)
print("r2 score:",r2_score(y_test,y_prediction))