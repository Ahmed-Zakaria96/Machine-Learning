import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import shapiro
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

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
        # columns to be dropped
        CTBD = null_df[null_df['Null Count']/m >= self.null_threshold]
        # rows to be dropped
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

        # delete records from column with value < .06
        cols = [s[0] for s in RTBD.values]
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
            if n in numCols:
                numNull = np.append(numNull, n)
            else:
                catNull = np.append(catNull, n)

        # fill numerical
        self.data[numNull] = self.data[numNull].apply(lambda x: x.fillna(x.mean()))
        # fill categorical
        self.data[catNull] = self.data[catNull].apply(lambda x: x.fillna(x.mode()[0]))

    # duplicated
    def handleDuplicates(self):
        # rows, columns
        m, n = self.data.shape
        # list of columns with same value
        dupCol = []
        for c, cData in self.data.iteritems():
            # Value counts
            VC = any(cData.value_counts().values/m >.7)
            if VC:
                dupCol.append(c)
        # drop columns with mostly same value
        self.data.drop(columns=dupCol, inplace=True)

    # correlated features
    def handleCorrFeature(self):
        numCols = [c for c in self.data.columns.tolist() if c in self.grabNumeric()]
        CM = self.data[numCols].corr()

        redundantFeatures = []
        corrValues = []
        for index, i in enumerate(numCols):
            if i == self.target or i == self.skip:
                continue
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


    # hand outliers
    def handleOutliers(self):
        numCols = [c for c in self.data.columns.tolist() if c in self.grabNumeric()]
        for c in self.data[numCols]:
            Q1 = self.data[c].quantile(.25)
            Q3 = self.data[c].quantile(.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            LO = self.data.index[self.data[c] < lower].tolist()
            UO = self.data.index[self.data[c] > upper].tolist()

            if len(LO) > 0:
                self.data.loc[LO, c] = lower

            if len(UO) > 0:
                self.data.loc[UO, c] = upper

    # check skewness
    def calcSkew(self):
        n = self.data.shape[0]
        numCols = self.grabNumeric()
        numCols.remove(self.target)
        mu = self.data[numCols].mean()
        std = self.data[numCols].std()
        skw = pd.DataFrame(np.sum(np.power((self.data[numCols] - mu), 3)) / ((n - 1) * std) )
        return skw

    # log transformation for skewed features
    def handleSkew(self):
        skw = self.calcSkew()
        for s in skw.index.tolist():
            if skw.loc[s][0] > 1 or skw.loc[s][0] < -1:
                self.data[s] = np.log(1 + abs(self.data[s]))

    # check for normal distributed features
    def checkDistribution(self):
        numCols = self.grabNumeric()
        numCols.remove(self.target)

        gaussianFeatures = []
        nonGaussianFeatures = []
        for c in numCols:
            w_stat, p = shapiro(self.data[c])
            print('W_Statistic=%.3f, p=%.8f' % (w_stat, p))

            if p > self.alpha:
                print(f'{c} looks like gaussian (fail to reject H0)')
                gaussianFeatures.append(c)
            else:
                print(f'{c} does not look Gaussian (reject H0)')
                nonGaussianFeatures.append(c)


        return gaussianFeatures, nonGaussianFeatures

    # draw QQ plot
    def drawQQ(self):
        numCols = self.grabNumeric()
        numCols.remove(self.target)
        nC = 4
        nR = len(numCols) // 4 if len(numCols) % 4 == 0 else (len(numCols) // 4) + 1
        if nR == 1:
            fig, axes = plt.subplots(nrows=nR, ncols=nC, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(nrows=nR, ncols=nC, figsize=(30, 30))
        gaussianFeatures, nonGaussianFeatures = self.checkDistribution()
        i=0
        j=0
        for col in nonGaussianFeatures:
            if nR == 1:
                sm.qqplot(self.data[col],fit = False, line='q', ax = axes[j])
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


    # normalize data
    def skLR(self):
        numCols = self.grabNumeric()
        numCols.remove(self.target)
        scaler = StandardScaler()
        xTrain, xTest, yTrain, yTest = train_test_split(self.data[numCols].to_numpy(),
                                                self.data[self.target].to_numpy(),
                                                test_size=.2, random_state=42)

        scaler = scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)
        LR = LinearRegression()
        LR.fit(xTrain, yTrain)
        print('Accuracy of LR on training set: {:.2f}'
             .format(LR.score(xTrain, yTrain)))

        print('Accuracy of LR on test set: {:.2f}'
             .format(LR.score(xTest, yTest)))
        return xTrain, yTrain
