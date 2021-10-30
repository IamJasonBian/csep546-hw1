import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

os.chdir('C:\\Users\\Jason\\Desktop\\csep546-hw1\\homeworks\\poly_regression')

from polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """
    
    os.chdir("C:/Users/Jason/Desktop/csep546-hw1/")
    # load the data
    filePath = "data/polyreg/polydata.dat"
    file = open(filePath, "r")
    allData = np.loadtxt(file, delimiter=",")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, reg_lambda=4)
    model.fit(X, y)

    print(model.fit)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d}")
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
