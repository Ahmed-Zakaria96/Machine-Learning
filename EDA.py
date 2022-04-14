import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class EDA:
    # numeric data types
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def __init__(self, data, target, skip=None, null_threshold=.6, dup_threshold=.8, corr_threshold=.7, alpha=.05):
        self.data = data
        self.null_threshold = null_threshold
        self.dup_threshold = dup_threshold
        self.corr_threshold = corr_threshold
        self.alpha = alpha
        self.target = target
        self.skip = skip


    # grab numerical data
    def grabNumeric(self):
        return list(set(self.data.select_dtypes(include=self.numerics).columns) - set(["Id"]))

    # grab categorical data
    def grabCategorical(self):
        return list(set(self.data.select_dtypes(include=['object']).columns))

    # define nulls
    def grabNulls(self):
        m = self.data.shape[0]
        null_df = self.data.isna().sum().reset_index().rename(columns={0: "Null Count"}).sort_values(by=['Null Count'], ascending=False)
        null_df = null_df[null_df["Null Count"] > 0]
        # columns to be dropped > null_threshold
        CTBD = null_df[null_df['Null Count']/m >= self.null_threshold]
        # rows to be dropped < .1 of samples
        RTBD = null_df[null_df['Null Count']/m <= .1]
        # Records to be filled
        RTBF = null_df[((null_df['Null Count']/m <= self.null_threshold)
                      & (null_df['Null Count']/m > .1))]

        return CTBD, RTBD, RTBF

    def handleNulls(self):
        CTBD, RTBD, RTBF = self.grabNulls()
        # drop columns with nulls > threshold
        nCols = [s[0] for s in CTBD.values]
        self.data = self.data.drop(columns=nCols)

        # grab cols with rows cotaining nulls in it
        cols = [s[0] for s in RTBD.values]
        # delete records from column with value < .06
        self.data = self.data.dropna(subset=cols)

        # fill records with mean
        # seperate numeric cols from categorical
        numCols = self.grabNumeric()
        catCols = self.grabCategorical()
        # nurical cols to be filled
        numNull = np.array([])
        # categorical cols to be filled
        catNull = np.array([])
        N = [s[0] for s in RTBF.values]
        for n in N:
            # filter null based on colum type numerical or categorical
            if n in numCols:
                numNull = np.append(numNull, n)
            else:
                catNull = np.append(catNull, n)

        # fill numerical cols with mean
        self.data[numNull] = self.data[numNull].apply(lambda x: x.fillna(x.mean()))
        # fill categorical cols with mod
        self.data[catNull] = self.data[catNull].apply(lambda x: x.fillna(x.mode()[0]))

    # duplicated
    def handleDuplicates(self):
        # rows, columns
        m, n = self.data.shape
        # list of columns with same value
        dupCol = []
        for c, cData in self.data.iteritems():
            # Value counts
            VC = any(cData.value_counts().values/m > self.dup_threshold)
            if VC:
                dupCol.append(c)
        # drop columns with mostly same value
        self.data.drop(columns=dupCol, inplace=True)
        return dupCol

    # correlated features
    def handleCorrFeature(self):
        numCols = [c for c in self.data.columns.tolist() if c in self.grabNumeric()]
        CM = self.data[numCols].corr()
        # features to be deleted
        redundantFeatures = []
        # correlation values
        corrValues = []
        for index, i in enumerate(numCols):
            # skip target column in the filtering or other custom table
            if i == self.target or i == self.skip:
                continue
            # loop over the upper triangle matrix of the corr matrix since it is symetric
            for j in numCols[index+1:-1]:
                if j == self.skip:
                    continue
                # correlation between 2 features
                cSample = abs(CM.loc[i][j])

                # check for correlation threshold
                if cSample >= .75:
                    # choose which feature is more correlated to target
                    if abs(CM.loc[i][self.target]) > abs(CM.loc[j][self.target]):
                        redundantFeatures.append(j)

                    else:
                        redundantFeatures.append(i)

                    corrValues.append({
                        "Feature correlation":  CM.loc[i][j],
                        f"Feature {i} vs {self.target}":  CM.loc[i][self.target],
                        f"Feature {j} vs {self.target}":  CM.loc[j][self.target],
                    })
        # drop redundant features
        self.data.drop(columns=redundantFeatures, inplace=True)
        return redundantFeatures, corrValues


    def checkOutliers(self):
        numCols = [c for c in self.data.columns.tolist() if c in self.grabNumeric()]
        # dict to hold outliers
        outliers = {}
        for c in self.data[numCols]:
            Q1 = self.data[c].quantile(.25)
            Q3 = self.data[c].quantile(.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # grab rows < lower bound
            LO = self.data.index[self.data[c] < lower].tolist()
            # grab rows > upper bound
            UO = self.data.index[self.data[c] > upper].tolist()

            outliers[c] = {
                "Lower Bound": lower,
                "Below Lower": LO,
                "Upper Bound": upper,
                "Above Upper": UO
            }

        return outliers

    # box plot outliers
    def boxplotOutliers(self):
        outliers = list(self.checkOutliers().keys())
        j = 0
        nC = 6
        nR = len(outliers) // nC if len(outliers) % nC == 0 else (len(outliers) // nC) + 1
        if nR == 1:
            fig, axes = plt.subplots(nrows=nR, figsize=(20, 10))
            sns.boxplot(data=self.data[outliers])

        else:
            fig, axes = plt.subplots(nrows=nR, figsize=(30, 30))
            for i in range(0, len(outliers), nC):
                sns.boxplot(data=self.data[outliers[i:i+nC]], ax=axes[j])
                j += 1


    # hand outliers
    def handleOutliers(self):
        # grab the outliers
        outliers = self.checkOutliers()

        for c in outliers:
            # grab col
            col = outliers[c]
            # if there are values below lower bound
            if len(col['Below Lower']) > 0:
                # replace them with the lower bound
                self.data.loc[col['Below Lower'], c] = col['Lower Bound']
            # if there are values above upper bound
            if len(col['Above Upper']) > 0:
                # replace with the upper bound
                self.data.loc[col['Above Upper'], c] = col['Upper Bound']

    # check skewness
    def calcSkew(self):
        n = self.data.shape[0]
        numCols = self.grabNumeric()
        mu = self.data[numCols].mean()
        std = self.data[numCols].std()
        skw = pd.DataFrame(np.sum(np.power((self.data[numCols] - mu), 3)) / ((n - 1) * std) ).rename(columns={0: "Skew Value"})
        return skw

    # log transformation for skewed features
    def handleSkew(self):
        skw = self.calcSkew()
        for s in skw.index.tolist():
            if skw.loc[s][0] > 1 or skw.loc[s][0] < -1:
                # aplly log transform to column with abs(skewness) > 1 (+, -)
                self.data[s] = np.log(1 + abs(self.data[s]))

    # check for normal distributed features
    def checkDistribution(self):
        numCols = self.grabNumeric()

        # list for gaussianFeatures
        gaussianFeatures = []
        # list for nonGaussianFeatures
        nonGaussianFeatures = []
        for c in numCols:
            # calc w and p Statistics for each column
            w_stat, p = shapiro(self.data[c])
            print('W_Statistic=%.3f, p=%.8f' % (w_stat, p))

            # if p > alpha add to gaussianFeatures
            if p > self.alpha:
                print(f'{c} looks like gaussian (fail to reject H0)')
                gaussianFeatures.append(c)

            # if p < alpha add to nongaussianFeatures
            else:
                print(f'{c} does not look Gaussian (reject H0)')
                nonGaussianFeatures.append(c)

        return gaussianFeatures, nonGaussianFeatures


    # scale features
    def featureScale(self):
        gFeatures, nonGFeatures = self.checkDistribution()
        # std scale gausian features
        if len(gFeatures) > 0:
            stdScaler = StandardScaler()
            stdScaler = stdScaler.fit(self.data[gFeatures])
            self.data[gFeatures] = stdScaler.transform(self.data[gFeatures])

        # minmax scale non gausian features
        if len(nonGFeatures) > 0:
            mmScaler = MinMaxScaler()
            mmScaler = mmScaler.fit(self.data[nonGFeatures])
            self.data[nonGFeatures] = mmScaler.transform(self.data[nonGFeatures])

    # draw QQ plot
    def drawQQ(self):
        numCols = self.grabNumeric()
        if self.target in numCols:
            numCols.remove(self.target)
        nC = 4
        nR = len(numCols) // 4 if len(numCols) % 4 == 0 else (len(numCols) // 4) + 1
        if nR == 1:
            fig, axes = plt.subplots(nrows=nR, ncols=len(numCols), figsize=(20, 10))
        else:
            fig, axes = plt.subplots(nrows=nR, ncols=nC, figsize=(50, 100))

        i=0
        j=0
        for col in numCols:
            if nR == 1:
                sm.qqplot(self.data[col],fit = False, line='q', ax = axes[j])
                axes[j].set_title(col)
                if(j<nC-1):
                    j+=1
                else:
                    i+=1
                    j=0
            else:
                sm.qqplot(self.data[col],fit = False, line='q', ax = axes[i, j])
                axes[i, j].set_title(col)
                if(j<nC-1):
                    j+=1
                else:
                    i+=1
                    j=0
        plt.show();


    #split data
    def trainTestSplit(self, test_size, random_state, include=None, exclude=None):
        if include is None and exclude is None:
            numCols = self.grabNumeric()
            if self.target in numCols:
                numCols.remove(self.target)
        elif include is not None:
            numCols = include
        elif exclude is not None:
            numCols = self.grabNumeric()
            numCols.remove(exclude)
            if self.target in numCols:
                numCols.remove(self.target)
        else:
            numCols = self.grabNumeric()
            if self.target in numCols:
                numCols.remove(self.target)

        xTrain, xTest, yTrain, yTest = train_test_split(self.data[numCols],
                                                self.data[self.target],
                                                test_size=test_size, random_state=random_state)

        return xTrain, xTest, yTrain, yTest
