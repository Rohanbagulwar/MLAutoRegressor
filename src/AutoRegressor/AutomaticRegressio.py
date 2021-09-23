import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AutomaticRegression:

    def __init__(self, features, label, model=None):
        """
        :param features:
        :param label:
        :param model:
        """
        self.features = features
        self.label = label
        self.nanFrame = None
        self.numericalFilled = pd.DataFrame()
        self.categoricalFilled = pd.DataFrame()
        self.cleanedData = None
        self.__scaler = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__model = None
        self.__y_pred = None

        error = self.__validater()
        if error is None:
            self.autoMl(model)

    def __validater(self):
        """

        :return:
        """
        if self.label is None:
            raise ValueError("Label Not Provided")

        if self.features is None:
            raise ValueError("Features Not Provided")

    def autoMl(self, model):
        """

        :param model:
        :return:
        """
        if model is None:
            self.__nanValues()
            self.__seperateFeatures()

            return None
        else:
            self.__nanValues()
            self.__seperateFeatures()
            self.autoNanFill()
            self.scaling()
            self.trainModel(model)
            self.pickleFile()

    def __nanValues(self):
        """

        :return:
        """
        total = len(self.features)
        frame = self.features
        frame = frame.isnull().sum()
        frame = frame.reset_index()
        frame.columns = ['Columns', 'Nan']
        frame['NanPercent'] = (frame['Nan'] / total)
        frame = frame[frame['Nan'] > 0].sort_values('Nan')
        self.nanFrame = frame
        return frame

    def __seperateFeatures(self):
        """

        :return:
        """
        self.NumericalFeatures = self.features.select_dtypes(exclude='object')
        self.CategoricalFeatures = self.features.select_dtypes(include='object')
        return self.NumericalFeatures, self.CategoricalFeatures

    def autoNanFill(self):
        """

        :return:
        """
        for i in self.NumericalFeatures.columns:
            self.numericalFilled[i] = self.NumericalFeatures[i].fillna(self.NumericalFeatures[i].mean())

        for i in self.CategoricalFeatures.columns:
            self.categoricalFilled[i] = self.CategoricalFeatures[i].fillna(self.CategoricalFeatures[i].mode()[0])

        self.cleanedData = pd.concat([self.categoricalFilled, self.numericalFilled], axis=1)
        return self.cleanedData

    def manualFeatureSelection(self, columns=None):
        """

        :param columns:
        :return:
        """
        if type(columns) == list:
            self.cleanedData = self.cleanedData[columns]
            return self.cleanedData
        else:
            return 'Please provide list of columns'

    # Model Training
    def __oneHotEncoding(self, features):
        """
        :param features:
        :return:
        """
        self.__oneHotData = pd.get_dummies(features, drop_first=True)
        return self.__oneHotData

    def __train_test_Split(self):
        """

        :return:
        """

        self.cleanedData = self.__oneHotEncoding(self.cleanedData)
        X_train, X_test, y_train, y_test = train_test_split(self.cleanedData,
                                                            self.label,
                                                            test_size=0.30,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def scaling(self, typ='Standard'):
        """

        :param typ:
        :return:
        """

        if typ == 'Standard':
            self.__scaler = StandardScaler()

            X_train, X_test, y_train, y_test = self.__train_test_Split()

            X_train = self.__scaler.fit_transform(X_train)
            X_test = self.__scaler.transform(X_test)

            self.__X_train = X_train
            self.__X_test = X_test
            self.__y_train = y_train
            self.__y_test = y_test

            self.pickleScaler(self.__scaler)

            return self.__X_train, self.__X_test, self.__y_train, self.__y_test
        else:
            self.__scaler = MinMaxScaler()

            X_train, X_test, y_train, y_test = self.__train_test_Split()

            X_train = self.__scaler.fit_transform(X_train)
            X_test = self.__scaler.transform(X_test)

            self.__X_train = X_train
            self.__X_test = X_test
            self.__y_train = y_train
            self.__y_test = y_test

            self.pickleScaler(self.__scaler)

            return self.__X_train, self.__X_test, self.__y_train, self.__y_test

    @staticmethod
    def __accuracy(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        MAE = mean_absolute_error(y_true, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        R2 = r2_score(y_true, y_pred)

        return MAE, RMSE, R2

    def trainModel(self, model):
        """

        :param model:
        :return:
        """
        self.__model = model
        self.__model.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__model.predict(self.__X_test)

        MAE, RMSE, R2 = self.__accuracy(self.__y_test, self.__y_pred)
        print('Mean_absolute_error :', MAE)
        print('Root_Mean_squared_error :', RMSE)
        print('R2_Score: ', R2)

    def pickleFile(self, model=None, filename='ModelCreatedByAutoMl'):
        """

        :param model:
        :param filename:
        :return:
        """
        if model is None:
            joblib.dump(self.__model, filename)
        else:
            joblib.dump(model, filename)

    def pickleScaler(self, scaler=None, filename='ScalerCreatedByAutoMl'):
        """

        :param scaler:
        :param filename:
        :return:
        """
        if scaler is None:
            joblib.dump(self.__scaler, filename)
        else:
            joblib.dump(scaler, filename)
