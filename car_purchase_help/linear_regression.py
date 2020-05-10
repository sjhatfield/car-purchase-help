from sklearn.linear_model import LinearRegression
import numpy as np


class Linear_Regression:
    """
    Custom class that contains the sklearn linear regression as well
    as the training data, predicitons for training X and mean absolute
    residual which are needed to give advice. This means predicitons 
    can be given without loading the original dataframe
    """

    def __init__(self) -> None:
        """ 
        Constructor. Just initializes the linear regressor
        """
        self.linreg = LinearRegression()

    def fit(self, Xtrain, ytrain) -> None:
        """
        Fits the linear regression.
        :param X_train: single column of pandas dataframe (so series) should be `odometer`
        :param y_train: single column of pandas dataframe should be `price`
        """
        # Save the training data
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        # Fit the regression
        self.linreg.fit(X=Xtrain, y=ytrain)
        # Save the predicitons for the training X and mean absolute residual
        self.y_preds = self.linreg.predict(np.sort(Xtrain, axis=0))
        self.mean_absolute_residual = np.mean(abs(self.y_preds - ytrain))

    def predict(self, x: float) -> float:
        """
        Given single odometer value return the price prediciton
        :param x: float odometer value
        :return: the price prediciton
        """
        return self.linreg.predict(X=[[x]])[0][0]

    # All the getters
    def get_Xtrain(self):
        return self.Xtrain

    def get_ytrain(self):
        return self.ytrain

    def get_y_preds(self):
        return self.y_preds

    def get_mean_absolute_residual(self):
        return self.mean_absolute_residual
