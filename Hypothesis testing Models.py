import pandas as pd
import numpy as np
from scipy import stats



datasets={'caffe','incubator-mxnet','keras','pytorch','tensorflow'}


#Wilcoxon signed rank test for all performance metrics between the baseline and 1 phase MNB tool
for i in datasets:

    df1 = pd.read_csv(f'Results/1Phase_{i}_GaussianNB_RAWDATA.csv').fillna("")
    df2 = pd.read_csv(f'Results/1Phase_{i}_MultinomialNB_RAWDATA.csv').fillna("")

    cols = df1.columns
    reject = True
    print(f"\n###{i}###")
    for i in cols[1:-1]:
        test1 = df1[i].tolist()
        test2 = df2[i].tolist()

        t_stat, p_value = stats.wilcoxon(test1,test2)
    
        alpha = 0.05
    
        #print(f"\ncompareing {i}")
        if p_value < alpha:
            #print(f"{i} Reject null, Significant difference between performance")
            None
        else:
            print(f"{i} accept, no significant difference in model performance")
            reject = False
        #print(t_stat)
        #print(p_value)
    if reject == False:
        print("we cannot fully reject null model as being better as it outperforms in at least one metric")
    
    else:
        print("We can fully reject null model, as it is surpassed in all criteria")


#Welch T-Test for all performance metrics between the 2-Phase MNB and 1-Phase MNB solutions
for i in datasets:

    df1 = pd.read_csv(f'Results/2Phase_{i}_MultinomialNB_RAWDATA.csv').fillna("")
    df2 = pd.read_csv(f'Results/1Phase_{i}_MultinomialNB_RAWDATA.csv').fillna("")

    cols = df1.columns
    reject = True
    print(f"\n###{i}###")
    for i in cols[1:-1]:
        test1 = df1[i].tolist()
        test2 = df2[i].tolist()

        t_stat, p_value = stats.ttest_rel(test1,test2)
    
        alpha = 0.05
    
        #print(f"\ncompareing {i}")
        if p_value < alpha:
            #print(f"{i} Reject null, Significant difference between performance")
            None
        else:
            print(f"{i} accept, no significant difference in model performance")
            reject = False
        #print(t_stat)
        #print(p_value)
    if reject == False:
        print("we cannot fully reject null model as being better as it outperforms in at least one metric")
    
    else:
        print("We can fully reject null model, as it is surpassed in all criteria")


