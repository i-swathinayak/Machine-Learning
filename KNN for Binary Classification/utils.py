import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    f1_sc = (2 * sum(x * y for x, y in zip(real_labels, predicted_labels))) / (sum(real_labels) + sum(predicted_labels))
    return f1_sc


real_labels=[1,1,0,1,0,1]
predicted_labels=[1,0,1,0,1,1]
print(f1_score(real_labels, predicted_labels))

class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        x=np.array(point1)
        y=np.array(point2)
        return np.sum(abs(x-y)/(abs(x) + abs(y)))


    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        x=np.array(point1)
        y=np.array(point2)
        return np.cbrt(np.sum(abs(x-y)**3))


    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x=np.array(point1)
        y=np.array(point2)
        return np.sqrt(np.sum((x-y)**2))


    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x=np.array(point1)
        y=np.array(point2)
        return np.inner(x, y)


    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x=np.array(point1)
        y=np.array(point2)
        cosine_similarity = np.inner(x, y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))
        return 1-cosine_similarity
    

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x=np.array(point1)
        y=np.array(point2)
        return -np.exp(-0.5 * np.sum((x-y)**2))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        distance_tie={'cosine_dist': 1, 'inner_prod': 2, 'gaussian': 3, 'euclidean': 4, 'minkowski': 5, 'canberra':6}
        opt_score = float('-inf')
        opt_k = np.array(x_train).shape[0] - 1

        if opt_k > 30:
            opt_k = 30
        for k in range(1, opt_k, 2):
            for key, val in distance_funcs.items():
                knn = KNN(k, val)
                knn.train(x_train, y_train)
                score = f1_score(y_val, knn.predict(x_val))
                if opt_score < score:
                    opt_score = score
                    self.best_model = knn
                    self.best_k = k
                    self.best_distance_function = key
                elif opt_score == score and distance_tie.get(key) > distance_tie.get(self.best_distance_function):
                    self.best_model = knn
                    self.best_k = k
                    self.best_distance_function = key


# TODO: find parameters with the best f1 score on validation dataset, with normalized data
def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
    """
    This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
    tune k and disrance function, you need to create the normalized data using these two scalers to transform your
    data, both training and validation. Again, we will use f1-score to compare different models.
    Here we have 3 hyperparameters i.e. k, distance_function and scaler.

    :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
        loop over all distance function for each data point and each k value.
        You can refer to test.py file to see the format in which these functions will be
        passed by the grading script
    :param scaling_classes: dictionary of scalers you will use to normalized your data.
    Refer to test.py file to check the format.
    :param x_train: List[List[int]] training data set to train your KNN model
    :param y_train: List[int] train labels to train your KNN model
    :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
        labels and tune your k, distance function and scaler.
    :param y_val: List[int] validation labels

    Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
    self.best_distance_function, self.best_scaler and self.best_model respectively

    NOTE: When there is a tie, choose model based on the following priorities:
    For normalization, [min_max_scale > normalize];
    Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
    If they have same distance function, choose model which has a less k.
    """

    # You need to assign the final values to these variables
    self.best_k = None
    self.best_distance_function = None
    self.best_scaler = None
    self.best_model = None

    distance_tie = {'cosine_dist': 1, 'inner_prod': 2, 'gaussian': 3, 'euclidean': 4, 'minkowski': 5, 'canberra': 6}
    scaler_tie={'normalize': 1, 'min_max_scale': 2}

    opt_score=float('-inf')
    opt_k=np.array(x_train).shape[0]-1
    if opt_k>30:
        opt_k=30
    for scaler_key, scaler_val in scaling_classes.items():
        scaler = scaler_val()
        train_data = scaler(x_train)
        for k in range(1,opt_k,2):
            for key, val in distance_funcs.items():
                knn=KNN(k,val)
                knn.train(train_data,y_train)
                score=f1_score(y_val,knn.predict(scaler(x_val)))
                if opt_score < score:
                    opt_score=score
                    self.best_model=knn
                    self.best_k=k
                    self.best_distance_function=key
                    self.best_scaler=scaler_key
                elif opt_score == score and (scaler_tie.get(scaler_key) > self.best_scaler or distance_tie.get(key) > distance_tie.get(self.best_distance_function)):
                    self.best_k = k
                    self.best_distance_function = key
                    self.best_scaler = scaler_key

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        data = np.array(features,dtype=float)
        for i in range(data.shape[0]):
            norm_factor=np.sqrt(sum([x*x for x in data[i]]))
            if norm_factor!=0:
                for j in range(data.shape[1]):
                    data[i][j]=data[i][j]/norm_factor
        return data.tolist()


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_call=False
        self.min=float('inf')
        self.max=float('-inf')

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        data = np.array(features, dtype=float)

        if self.first_call == False:
            self.first_call = True
            self.min = np.amin(data, axis=0)
            self.max = np.amax(data, axis=0)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if self.max[j] == self.min[j]:
                    data[i][j] = 0.0
                else:
                    data[i][j] = (data[i][j] - self.min[j]) / (self.max[j] - self.min[j])

        return data.tolist()
















