import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics

def random_forest_regressor (X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 1)
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    prediction = forest_model.predict(val_X)
    mae = mean_absolute_error(prediction, val_y)
    print(f"\nRF Mean Absolute Error: {mae:.4f}")
    total_mean_error = (mae * 100) / y.mean()
    print(f"RF Total Mean Error: {total_mean_error:.4f} %\n")
    return(mae, prediction)

def decision_tree_regressor(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 1)
    
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        prediction = model.predict(val_X)
        mae = mean_absolute_error(val_y, prediction)
        return(mae)
    
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    for max_leaf_nodes in candidate_max_leaf_nodes:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

    # Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
    scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
    best_tree_size = min(scores, key=scores.get)
    print(f"Best tree size: {best_tree_size}")
    mae = scores.get(best_tree_size)
    final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

    # fit the final model and uncomment the next two lines
    final_model.fit(val_X, val_y)
    prediction = final_model.predict(val_X)

    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_X, train_y)
    print(f"\nDT Mean Absolute Error: {mae:.4f}")
    total_mean_error = (mae * 100) / y.mean()
    print(f"DT Total mean error: {total_mean_error:.4f} %")
    return(mae, prediction)
    