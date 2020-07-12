import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("weather.csv")
print(data)
 
  

data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches',  
                  'SeaLevelPressureLowInches'], axis = 1) 
  

data = data.replace('T', 0.0) 
  

data = data.replace('-', 0.0) 
  
 
data.to_csv('weather_final.csv') 

data = pd.read_csv("weather_final.csv") 


X = data.drop(['PrecipitationSumInches'], axis = 1) 


Y = data['PrecipitationSumInches'] 

Y = Y.values.reshape(-1, 1) 


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state= 45)
 

day_index = 500   #722 is high
days = [i for i in range(Y.size)] 

clf = LinearRegression() 

clf.fit(X, Y) 

Y_predict=clf.predict(X_test)

inp = np.array([[53], [60], [35], [67], [75], [43], [33], [45], 
				[57], [49.68], [10], [7], [2], [0], [32], [44], [22]]) 


inp = inp.reshape(1, -1) 

print('The precipitation in inches for the input is:', clf.predict(inp))


 
print("The precipitation trend graph: ") 
plt.scatter(days, Y, color = 'g') 
plt.scatter(days[day_index], Y[day_index], color ='r') 
plt.title("Precipitation level") 
plt.xlabel("Days") 
plt.ylabel("Precipitation in inches") 


plt.show() 


print("Accuracy=", r2_score(Y_test, Y_predict))
