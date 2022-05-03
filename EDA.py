from scipy.stats import shapiro
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class EDA:
    # numeric data types
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def __init__(self, train_data, test_data, target, skip=None, null_threshold=.6, dup_threshold=.8, corr_threshold=.7, alpha=.05):
        self.train_data = train_data
        self.test_data = test_data
        self.null_threshold = null_threshold
        self.dup_threshold = dup_threshold
        self.corr_threshold = corr_threshold
        self.alpha = alpha
        self.target = target
        self.skip = skip

    # grab numerical data
    def grabNumeric(self, target=True, skip=True):
        numCols = list(set(self.train_data.select_dtypes(include=self.numerics).columns) - set(["Id"]))
        if target == False:
            numCols.remove(self.target)
        if self.skip is not None and skip == True:
            numCols = list(set(numCols) - set(self.skip))
        return numCols

    # grab categorical data
    def grabCategorical(self):
        return list(set(self.train_data.select_dtypes(include=['object']).columns))

    def visualize(self, plot=None):

        # distribution plot
        if plot == 'dist' or plot is None:
            numCols = self.grabNumeric()
            nR = len(numCols) // 4 if len(numCols) % 4 == 0 else len(numCols) // 4 +1
            fig, axes = plt.subplots(nrows=nR, ncols=4, figsize=(20, nR*6))
            i = 0
            j = 0
            for c in numCols:
                if nR == 1:
                    sns.histplot(x=self.train_data[c], ax=axes[i])
                    i += 1
                else:
                    sns.histplot(x=self.train_data[c], ax=axes[i, j])
                    if j < 3:
                        j +=1
                    else:
                        i += 1
                        j = 0

            fig.suptitle('Distribution of numerical features', fontsize=24, color='darkred')
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

        # boxplot numerical data
        if plot == 'boxplot' or plot is None:
            numCols = self.grabNumeric()
            nR = len(numCols) // 4 if len(numCols) % 4 == 0 else len(numCols) // 4 +1
            fig, axes = plt.subplots(nrows=nR, ncols=4, figsize=(20, nR*6))
            i = 0
            j = 0
            for c in numCols:
                if nR == 1:
                    sns.boxplot(x=self.train_data[c], orient='h', ax=axes[i])
                    i += 1
                else:
                    sns.boxplot(x=self.train_data[c], orient='h', ax=axes[i, j])
                    if j < 3:
                        j +=1
                    else:
                        i += 1
                        j = 0

            fig.suptitle('Boxplot of numerical features', fontsize=24, color='darkred')
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)


        # count plot categorical data
        if plot == 'countplot' or plot is None:
            catCols = self.grabCategorical()
            skipped = []
            if len(catCols) > 0:
                nR = len(catCols) // 4 if len(catCols) % 4 == 0 else len(catCols) // 4 + 1
                fig, axes = plt.subplots(nrows=nR, ncols=4, figsize=(20, nR*6))
                i = 0
                j = 0
                for c in catCols:
                    x = self.train_data[c].value_counts().index
                    y = self.train_data[c].value_counts().values
                    if len(x) > 100:
                        skipped.append(c)
                        continue

                    if nR == 1:
                        sns.barplot(x=x, y=y, data=self.train_data, ax=axes[i])
                        i += 1
                    else:
                        sns.barplot(x=x, y=y, data=self.train_data, ax=axes[i, j])
                        if j < 3:
                            j +=1
                        else:
                            i += 1
                            j = 0
            print("Skipped cols in count plot due to values > 100: \n", skipped)
            fig.suptitle('Countplot of categorical features', fontsize=24, color='darkred')
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

        # correlation heatmap
        if plot == 'corr' or plot is None:
            CM = self.train_data[self.grabNumeric()].corr()
            CM[(CM < 0.3) & (CM > -0.3)] = 0
            UCM = np.triu(np.ones_like(CM, dtype=bool))
            fig = plt.figure(figsize=(20, 15))
            ax = fig.add_subplot()
            sns.heatmap(data=CM, mask=UCM, ax=ax, annot=True)
            fig.suptitle('Correlation heat map of numerical features', fontsize=24, color='darkred')
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

    # define nulls
    def grabNulls(self, threshold=None):
        if threshold is not None:
            self.null_threshold = threshold
        m = self.train_data.shape[0]
        null_df = self.train_data.isna().sum().reset_index().rename(columns={0: "Null Count"}).sort_values(by=['Null Count'], ascending=False)
        null_df = null_df[null_df["Null Count"] > 0]
        # columns to be dropped > null_threshold
        CTBD = null_df[null_df['Null Count']/m >= self.null_threshold]
        # rows to be dropped < .1 of samples
        RTBD = null_df[null_df['Null Count']/m <= .1]
        # Records to be filled
        RTBF = null_df[((null_df['Null Count']/m <= self.null_threshold)
                      & (null_df['Null Count']/m > .1))]

        return CTBD, RTBD, RTBF

    def handleNulls(self, threshold=None):
        CTBD, RTBD, RTBF = self.grabNulls(threshold)
        # drop columns with nulls > threshold
        nCols = [s[0] for s in CTBD.values]
        self.train_data = self.train_data.drop(columns=nCols)
        # drop same columns from test data
        if self.test_data is not None:
            self.test_data.drop(columns=nCols, inplace=True)
            self.test_data.dropna(inplace=True)
        # grab cols with rows cotaining nulls in it
        cols = [s[0] for s in RTBD.values]
        # delete records from column with value < .06
        self.train_data = self.train_data.dropna(subset=cols)

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
        self.train_data[numNull] = self.train_data[numNull].apply(lambda x: x.fillna(x.mean()))
        # fill categorical cols with mod
        self.train_data[catNull] = self.train_data[catNull].apply(lambda x: x.fillna(x.mode()[0]))

    # duplicated
    def handleDuplicates(self, threshold=None):
        if threshold is not None:
            self.dup_threshold = threshold
        # rows, columns
        m, n = self.train_data.shape
        # list of columns with same value
        dupCol = []
        for c, cData in self.train_data.iteritems():
            # Value counts
            VC = any(cData.value_counts().values/m > self.dup_threshold)
            if VC:
                dupCol.append(c)

        self.train_data.drop(columns=dupCol, inplace=True)
        # drop same columns from test data
        if self.test_data is not None:
            self.test_data.drop(columns=dupCol, inplace=True)
        return dupCol

    # correlated features
    def handleCorrFeature(self, threshold=None):
        if threshold is not None:
            self.corr_threshold = threshold
        numCols = [c for c in self.train_data.columns.tolist() if c in self.grabNumeric()]
        CM = self.train_data[numCols].corr()
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
                if j == self.skip or j == self.target:
                    continue
                # correlation between 2 features
                cSample = abs(CM.loc[i][j])

                # check for correlation threshold
                if cSample >= self.corr_threshold:
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
        self.train_data.drop(columns=redundantFeatures, inplace=True)
        # drop same columns from test data
        if self.test_data is not None:
            self.test_data.drop(columns=redundantFeatures, inplace=True)
        return redundantFeatures, corrValues


    def checkOutliers(self, threshold=1.5):
        numCols = [c for c in self.train_data.columns.tolist() if c in self.grabNumeric(target=False)]
        # Q1 = self.train_data[numCols].quantile(.25)
        # Q3 = self.train_data[numCols].quantile(.75)
        # IQR = Q3 - Q1
        # lower = Q1 - threshold * IQR
        # upper = Q3 + threshold * IQR
        # lt_lower = self.train_data.index[self.train_data[numCols] < lower]
        # gt_upper = self.train_data.index[self.train_data[numCols] > lower]
        # bounds = np.c_[lower, lt_lower.sum(), upper, gt_upper.sum()]
        # outlier_df = pd.DataFrame(data=bounds, index=numCols, columns=['Lower Bound', 'Below Lower Count', 'Upper Bound', 'Above Upper Count'])
        # return outlier_df
        # dict to hold outliers
        outliers = {}
        for c in self.train_data[numCols]:
            Q1 = self.train_data[c].quantile(.25)
            Q3 = self.train_data[c].quantile(.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            # grab rows < lower bound
            LO = self.train_data.index[self.train_data[c] < lower].tolist()
            # grab rows > upper bound
            UO = self.train_data.index[self.train_data[c] > upper].tolist()

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
            sns.boxplot(data=self.train_data[outliers])

        else:
            fig, axes = plt.subplots(nrows=nR, figsize=(30, 30))
            for i in range(0, len(outliers), nC):
                sns.boxplot(data=self.train_data[outliers[i:i+nC]], ax=axes[j])
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
                self.train_data.loc[col['Below Lower'], c] = col['Lower Bound']
            # if there are values above upper bound
            if len(col['Above Upper']) > 0:
                # replace with the upper bound
                self.train_data.loc[col['Above Upper'], c] = col['Upper Bound']

    # check skewness
    def calcSkew(self, target):
        n = self.train_data.shape[0]
        numCols = self.grabNumeric(target=target)
        mu = self.train_data[numCols].mean()
        std = self.train_data[numCols].std()
        skw = pd.DataFrame(np.sum(np.power((self.train_data[numCols] - mu), 3)) / ((n - 1) * np.power(std, 3)) ).rename(columns={0: "Skew Value"})
        return skw

    # log transformation for skewed features
    def handleSkew(self, target=False):
        skw = self.calcSkew(target)
        for s in skw.index.tolist():
            if (skw.loc[s][0] > 1 or skw.loc[s][0] < -1) and skw.loc[s][0] >=0:
                # aplly log transform to column with abs(skewness) > 1 (+, -)
                self.train_data[s] = np.log(1 + abs(self.train_data[s]))
                if self.test_data is not None:
                    self.test_data[s] = np.log(1 + abs(self.test_data[s]))

    # check for normal distributed features
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
            fig, axes = plt.subplots(nrows=nR, ncols=nC, figsize=(20, nR*15))

        i=0
        j=0
        for col in numCols:
            if nR == 1:
                sm.qqplot(self.train_data[col],fit = False, line='q', ax = axes[j])
                axes[j].set_title(col)
                if(j<nC-1):
                    j+=1
                else:
                    i+=1
                    j=0
            else:
                sm.qqplot(self.train_data[col],fit = False, line='q', ax = axes[i, j])
                axes[i, j].set_title(col)
                if(j<nC-1):
                    j+=1
                else:
                    i+=1
                    j=0
        plt.show();

    # shapiro method
    def checkDistribution(self, threshold=None, target=True, skip=True):
        if threshold is not None:
            self.alpha = threshold
        numCols = self.grabNumeric(target=target, skip=skip)

        # list for gaussianFeatures
        gaussianFeatures = []
        # list for nonGaussianFeatures
        nonGaussianFeatures = []
        for c in numCols:
            # calc w and p Statistics for each column
            w_stat, p = shapiro(self.train_data[c].sample(n=500, replace=False))
            print('W_Statistic=%.3f, p=%.8f' % (w_stat, p))

            # if p > alpha add to gaussianFeatures
            if p >= self.alpha:
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
            stdScaler = stdScaler.fit(self.train_data[gFeatures])
            self.train_data[gFeatures] = stdScaler.transform(self.train_data[gFeatures])
            if self.test_data is not None:
                self.test_data[gFeatures] = stdScaler.transform(self.test_data[gFeatures])

        # minmax scale non gausian features
        if len(nonGFeatures) > 0:
            mmScaler = MinMaxScaler()
            mmScaler = mmScaler.fit(self.train_data[nonGFeatures])
            self.train_data[nonGFeatures] = mmScaler.transform(self.train_data[nonGFeatures])
            if self.test_data is not None:
                self.test_data[nonGFeatures] = mmScaler.transform(self.test_data[nonGFeatures])

    #split data
    def trainTestSplit(self, test_size, random_state, include=None, exclude=None):
        if include is None and exclude is None:
            numCols = self.grabNumeric(target=False)

        elif include is not None:
            numCols = include

        elif exclude is not None:
            numCols = self.grabNumeric(target=False)
            if exclude in numCols:
                numCols.remove(exclude)
        else:
            numCols = self.grabNumeric(target=False)

        if self.test_data is not None:
            xTrain = self.train_data[numCols]
            yTrain = self.train_data[self.target]
            xTest = self.test_data[numCols]
            yTest = self.test_data[self.target]
            # check gausian and non gausian features
            gFeatures, nonGFeatures = self.checkDistribution()

            if len(gFeatures) > 0:
                stdScaler = StandardScaler()
                stdScaler = stdScaler.fit(xTrain[gFeatures])
                xTrain = stdScaler.transform(xTrain[gFeatures])
                xTest = stdScaler.transform(xTest[gFeatures])

            if len(nonGFeatures) > 0:
                mmScaler = MinMaxScaler()
                mmScaler = mmScaler.fit(xTrain[nonGFeatures])
                xTrain = mmScaler.transform(xTrain[nonGFeatures])
                xTest = mmScaler.transform(xTest[nonGFeatures])

        else:
            xTrain, xTest, yTrain, yTest = train_test_split(self.train_data[numCols],
                                                    self.train_data[self.target],
                                                    test_size=test_size, random_state=random_state)

        return xTrain, xTest, yTrain, yTest
