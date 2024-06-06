import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle
from copy import deepcopy

def load_data(fname: pd.DataFrame):
    """
    load credit risk data and delete duplicates.
    functions also spew validation statement after dropping duplicates
    """
    data = pd.read_csv(fname)
    data_copy = deepcopy(data)
    data_copy.drop_duplicates(inplace=True)

    '''validation'''
    print(f"Data before dropping shape: {data.shape}")
    print(f"Data after dropping shape: {data_copy.shape}")

    return data

def split_input_output(data, target_col):
  """
  :param DataFrame data: targeted DataFrame
  :param str target_col: collumns to be dropped
  """
  y = data[target_col]
  X = data.drop(target_col, axis=1)

  # print out data shape
  print(f"(X) Input shape: {X.shape}")
  print(f"(y) Output shape: {y.shape}")

  return X, y

def split_train_test(input, output, test_size, seed, stratify):
  """
  split input data into two variables of train / test. valid set needs to be splitted again.
  :param input: input object
  :param output: target columns
  :param test_size: proportion of test size (defined in yaml file)
  :param seed: random state (CONSTANT for all step of model iteration)
  :param stratify: ensure same the distribution of classes as the raw data classes for return variables
  :return:
  """


  from sklearn.model_selection import train_test_split

  # first batch Train and temp
  X_train, X_temp, y_train, y_temp = train_test_split(input, output,
                                                              test_size=test_size,
                                                              random_state=seed,
                                                              stratify=stratify)
  ## train ratio
  list_class_train = y_train.value_counts().to_list()
  ratio_train = list_class_train[1] / list_class_train[0]

    ## train ratio
  list_class_temp = y_temp.value_counts().to_list()
  ratio_temp = list_class_temp[1] / list_class_temp[0]

  return X_train, X_temp, y_train, y_temp, ratio_train, ratio_temp

def serialize_data(data, path):
    """
    Save Data as pickle file
    :param data: Object to be dumped
    :param path: specified path for pickle data
    """
    with open(path, "wb") as file:
        pickle.dump(data, file)

def deserialize_data(path):
    """
    open serialized data
    :param path: specified path for pickle data
    :return data: file / data specified by the path
    """
    data = joblib.load(path)
    return data

def percent_missing_values(dataframe):
    """
    Calculate the percentage of missing values in each column of a DataFrame.
    """
    missing_value_info = []  # List to store missing value information

    for col in dataframe.columns:
        # catch missing value type na, None and ""
        missing_values = dataframe[col].isna().sum() + \
                         dataframe[col].eq(None).sum() + \
                         (dataframe[col] == '').sum()

        # calculate percentage of missing per colum
        percentage_missing = (missing_values / len(dataframe)) * 100
        print(f"{round(percentage_missing, 2)}% missing values of {col}") ; print("")

def drop_duplicate_data(X_data: pd.DataFrame, y_data: pd.Series):
    """
    Delete duplicates from X and Y
    :param X_data: Dataframe of input features
    :param y_data: Series of target columns

    :return:
    pd.DataFrame, Series: cleaned features and target columns
    """
    ## input validation
    if not isinstance(X_data, pd.DataFrame):
        raise ValueError("X_data has to be a DataFrame!")
    if not isinstance(y_data, pd.Series):
        raise ValueError("y_data has to be a DataFrame!")
    print("drop_duplicate_data: Parameters have been validated")

    X = X_data.copy() ; y = y_data.copy()
    print(f"\ndrop_duplicate_data: data shape before dropping : {X_data.shape}")

    X_duplicate = X[X.duplicated()]
    print(f"\ndrop_duplicate_data: duplicated data have the following shape : {X_duplicate.shape}")

    X_clean = (X.shape[0] - X_duplicate.shape[0], X.shape[1])
    print(f"\ndrop_duplicate_data: data shape have after dropping should be : {X_clean}")

    X.drop_duplicates(inplace=True)
    y = y.loc[X.index]

    print(f"\nData shape after dropping duplicates are: {X.shape}")

    return X, y

def median_imputation(data: pd.DataFrame, subset_data: pd.DataFrame, fit):

    if not isinstance(data, pd.DataFrame):
        raise RuntimeError("Fungsi median_imputation: parameter data haruslah bertipe DataFrame!")

    if fit == True:
        if not isinstance(subset_data, list):
            raise RuntimeError(
                "Fungsi median_imputation: untuk nilai parameter fit = True, subset_data harus bertipe list dan berisi daftar nama kolom yang ingin dicari nilai mediannya guna menjadi data imputasi pada kolom tersebut.")
    elif fit == False:
        if not isinstance(subset_data, dict):
            raise RuntimeError(
                "Fungsi median_imputation: untuk nilai parameter fit = False, subset_data harus bertipe dict dan berisi key yang merupakan nama kolom beserta value yang merupakan nilai median dari kolom tersebut.")
    else:
        raise RuntimeError(
            "Fungsi median_imputation: parameter fit haruslah bertipe boolean, bernilai True atau False.")

    print("Fungsi median_imputation: parameter telah divalidasi.")

    # Copy data
    data = data.copy()
    subset_data = deepcopy(subset_data)

    # Percabangan berdasarkan nilai fit
    if fit:
        imputation_data = {}
        for subset in subset_data:
            median_value = data[subset].median()
            imputation_data[subset] = median_value
        print(f"Fungsi median_imputation: proses fitting telah selesai, berikut hasilnya {imputation_data}.")
        return imputation_data
    else:
        print("Fungsi median_imputation: informasi count na sebelum dilakukan imputasi:")
        print(data.isna().sum())
        print("")

        data.fillna(subset_data, inplace=True)

        print("Fungsi median_imputation: informasi count na setelah dilakukan imputasi:")
        print(data.isna().sum())
        print("")

        return data







