# This program does regression analysis and some descriptive statistics.
# The data sets are x = total Covid test performed in Oregon Mar-Nov 2020.
# y = positive Covid test in Oregon Mar-Nov 2020.

import numpy as np
import matplotlib.pyplot as plt

# Function to compute the coefficients
def estimate_coef(x,y):

    # get the number of data points
    n = np.size(x)
    print("Number of Data Points is ", n)

    # mean of x and y vectors
    m_x, m_y = np.mean(x), np.mean(y)

    # calculation of cross-deviation and deviation about x
    SS_xy = np.sum(x*y)-n*m_y*m_x
    SS_xx = np.sum(x*x)-n*m_x*m_x

    # calculation of regression coefficients
    b_1 = SS_xy/SS_xx
    b_0 = m_y-b_1*m_x

    return(b_0,b_1)

# Function to compute the correlation Coefficient
def correlation_coefficient(x,y):
    c = np.corrcoef(x, y)
    return(c)

# Function to produce a regression plot
def plot_regression_line(x, y, b):

    # plotting a scatter plot
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting regression line
    plt.plot(x, y_pred, color = "g")

    # add labels
    plt.xlabel('x')
    plt.ylabel('y')

    # exhibit the plot
    plt.show()

# Function to generate a box plot of data
def box_plot(x):
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    labels = ['x1']
    ax1.boxplot(x, 0, 'gD',labels=labels)
    ax1.set_title('Box Plot')
    ax1.yaxis.grid(True)
    plt.show()

# Function to compute and print the Mean, and standard deviation of data sets
def descrip_stats(x, y):
    s = np.mean(x)
    t = np.std(x)
    print("Mean of x is {}", s)
    print("Standard Deviation of x is {}", t)
    s = np.mean(y)
    t = np.std(y)
    print("Mean of y is {}", s)
    print("Standard Deviation of y is {}", t)

# Function to read the data set files
def load_xdata() -> float:
    xdata = []
    # Read in the independent variables
    f = open("xDataFile.csv")
    for read_data in f.readlines():
        read_data = read_data.strip('\n')
        xdata.append(float(read_data))
    f.close
    print(xdata)
    return(xdata)

def load_ydata() -> float:
    ydata = []
    # Read in dependent variables
    f = open("yDataFile.csv")
    for read_data in f.readlines():
        read_data = read_data.strip('\n')
        ydata.append(float(read_data))
    f.close
    print(ydata)
    return(ydata)


def main():

    # Read in test observations
    a = load_xdata()
    b = load_ydata()
    print(a)
    x = np.array(a)
    y = np.array(b)

    # Call estimate coefficients function and print results
    b =estimate_coef(x, y)
    print("Estimated coefficients: \nb_0 = {} \ \nb_1 ={}".format(b[0], b[1]))

    # calculate correlation Pearsons correlation coefficient
    d = correlation_coefficient(x,y)
    print("Correlation Coefficient = ", d)
    descrip_stats(x, y)

    # Print boxplots of the data sets
    box_plot(x)
    box_plot(y)

    # output the regression line plot
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()





