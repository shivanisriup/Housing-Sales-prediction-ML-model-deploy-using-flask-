import pandas as pd
import pickle

data=pd.read_csv('housing.csv')

x=data[['square_feet']]
y=data[['price']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[500]]))