import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from functions import corr_analysis, random_forest_regressor, decision_tree_regressor
import matplotlib.pyplot as plt


# Path of the file to read
iowa_file_path = 'train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
df_pairplot = home_data[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SalePrice']]

maeRF, predictionRF = random_forest_regressor(X,y)
decision_tree_regressor(X,y)
print(home_data)
dictOfTuples = dict(corr_analysis(df_pairplot))

Y_dict = {(x,y):value for (x,y), value in dictOfTuples.items() if 'SalePrice' in (x,y)}

keys = [str(key) for key in Y_dict.keys()]
keys = [key for key in keys if keys.index(key) % 2 == 0]
values = [value for value in Y_dict.values()]
values = sorted(set([value for value in values]))

plot = plt.bar(keys,values)
plot = plt.xticks(rotation=90)
plot = plt.ylabel('pearson corr. value')
plot = plt.xlabel('column x column')
plt.show()