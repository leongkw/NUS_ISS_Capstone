import numpy as np
import pandas as pd

import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from typing import List

#######################################
## Function to reduce memory usage for pandas dataframe
#######################################
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


#######################################
## Function to rremove rare records below threshold | Default at 0.01% of distribution
#######################################
def remove_rare_records(df, col, threshold=0.0001):
    value_counts = df[col].value_counts(normalize=True)

    threshold = threshold
    to_keep = value_counts[value_counts >= threshold].index
    filtered_df = df[df[col].isin(to_keep)]

    return filtered_df

def association_analysis(df:pd.DataFrame, category_columns_to_test:List[str],
                         associate_only_single_feature_with_target:bool, target:str,
                         min_support:float, min_conf_threshold:float):

    '''
        Given a dataframe and a list of category columns, this function use the min_support
        and min_conf_threshold to either:
            1) Find the association of each feature with the target
            2) Find the association of all features

        Depending on the setting set by associate_only_single_feature_with_target.

        By default, the function does OHE, but does not drop first when doing it.
        By default, the function will fill NA, depending on the column type. Either int -999999 or str '-999999

        When doing association of all features, it is recommended to set a higher min_support and conf threshold
        to avoid memoryerror and crashing.

        Sample usage:

            association_analysis(
                df=df,
                category_columns_to_test=cat_features_to_test,
                associate_only_single_feature_with_target=True,
                target='HasDetections',
                min_support=0.01,
                min_conf_threshold=0.01

            )

    '''

    if associate_only_single_feature_with_target:
        dict_of_frequent_itemsets = {}
        dict_of_rules = {}
        for col in category_columns_to_test:
            temp_df = df[[col] + [target]]

            if pd.api.types.is_numeric_dtype(temp_df[col]):
                temp_df[col] = temp_df[col].fillna(-999999)
            else:
                temp_df[col] = temp_df[col].fillna('-999999')

            print(f'Unique values of {col}: \n\t {temp_df[col].unique()}')

            temp_df = pd.get_dummies(temp_df, columns=temp_df.columns, drop_first=False)

            frequent_itemsets = apriori(temp_df, min_support=min_support, use_colnames=True)

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf_threshold)

            print("Frequent Itemsets:")
            display(frequent_itemsets)
            print("\nAssociation Rules:")
            display(rules)
            print('\n\n=================\n\n')

            dict_of_frequent_itemsets[col] = frequent_itemsets
            dict_of_rules[col] = rules

            return dict_of_frequent_itemsets, dict_of_rules

    else:
        temp_df = df[category_columns_to_test + [target]]


        for col in category_columns_to_test:
            if pd.api.types.is_numeric_dtype(temp_df[col]):
                temp_df[col] = temp_df[col].fillna(-999999)
            else:
                temp_df[col] = temp_df[col].fillna('-999999')

        temp_df = pd.get_dummies(temp_df, columns=temp_df.columns, drop_first=False)

        frequent_itemsets = apriori(temp_df, min_support=min_support, use_colnames=True)

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf_threshold)

        print("Frequent Itemsets:")
        display(frequent_itemsets)
        print("\nAssociation Rules:")
        display(rules)
        print('\n\n=================\n\n')

        return frequent_itemsets, rules
