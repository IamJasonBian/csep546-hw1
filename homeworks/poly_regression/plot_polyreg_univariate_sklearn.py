import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import os


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """
    
    os.chdir('C:\\Users\\Jason\\Desktop\\csep546-hw1')

    # load the data
    filePath = "data/polyreg/polydata.dat"
    file = open(filePath, "r")
    allData = np.loadtxt(file, delimiter=",")

    x = allData[:, [0]]
    y = allData[:, [1]]
    
    polynomial_features= PolynomialFeatures(degree=8)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    print(rmse)
    print(r2)
    
    plt.scatter(x, y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()