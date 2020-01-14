'''
使用Kaggle提供的数据集进行信用评分分析
'''
import pandas as pd

#加载数据
data = pd.read_csv(r"C:\Users\Suface\Desktop\Python_code\Give_me_some_credit\Data\cs-training.csv",engine = "python")

#数据描述
pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行
print("\033[7;37;41m\t 数据详细描述： \033[0m")
print(data.describe())

data=data.iloc[:,1:]            #去掉第一列无用序号
print("\033[7;37;41m\t 数据简要描述： \033[0m")
print(data.info())               #查看数据集属性

# 用随机森林对缺失值进行预测（针对于缺失率比较大的变量MonthlyIncome）
from sklearn.ensemble import RandomForestRegressor

def set_missing(df):
    # 把已有的数值型特征取出来，MonthlyIncome位于第5列，NumberOfDependents位于第10列
    process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    X = known[:, 1:]    # X为特征属性值
    y = known[:, 0]    # y为结果标签值
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测月收入
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    # 用得到的预测结果填补原缺失数据
    df.loc[df.MonthlyIncome.isnull(), 'MonthlyIncome'] = predicted
    return df

#用随机森林填补比较多的缺失值
data=set_missing(data)

# #直接删除比较少的缺失值 ：NumberOfDependents，或者用下边的方法补全缺失值
data=data.dropna()

# #补全NumberOfDependents中的缺失值
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# print("\033[7;37;41m\t 查看NumberOfDependents中的情况： \033[0m")
# print(data.NumberOfDependents.value_counts())
# sns.countplot(x = 'NumberOfDependents', data = data)
# plt.show( )
#
# Dependents = pd.Series([0,1,2,3,4])     #用 0 1 2 3 4 代替缺失值
# for i in data['NumberOfDependents'][data['NumberOfDependents'].isnull()].index:
#     data['NumberOfDependents'][i] = Dependents.sample(1)

data['NumberOfDependents'][data['NumberOfDependents']>8] = 8          #家庭人口数超过8的全部用8代替（常识）

data = data.drop_duplicates()#删除重复项

print("\033[7;37;41m\t 展示填补完缺失值后的数据情况： \033[0m")
data.info()

#制作箱线图，观察数据是否存在异常
#这三个属性 NumberOfTime30-59DaysPastDueNotWorse逾期30-59天的次数,
          # NumberOfTimes90DaysLate贷款数量,
          # NumberOfTime60-89DaysPastDueNotWorse逾期60-89天的次数
import matplotlib.pyplot as plt
train_box = data.iloc[:,[3,7,9]]
train_box.boxplot(figsize=(10,4))
plt.show()
#观察到2个异常值点，删除，此外年龄应该大于0,小于100
data = data[data['age'] < 100]
data = data[data['age'] > 0]
data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]

data['SeriousDlqin2yrs'] = 1-data['SeriousDlqin2yrs']  # 使好客户为1 ， 坏客户为0，方便计数

#探索性分析，观察数据分布
import seaborn as sns
age=data['age']
sns.distplot(age)
plt.show()

mi=data[data['MonthlyIncome']<50000]['MonthlyIncome']
sns.distplot(mi)
plt.show()      #观察图，年龄和收入分布皆近似正态分布！）

#数据切割，将数据分为训练集和测试集：3-7
from sklearn.model_selection  import train_test_split
Y=data['SeriousDlqin2yrs']
X=data.ix[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
trainDf = pd.concat([Y_train, X_train], axis=1)
testDf = pd.concat([Y_test, X_test], axis=1)
clasTest = testDf.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()

#变量分箱（binning）
import numpy as np
import pandas as pd
from scipy import stats

#最优分段
def mono_bin(Y, X, n ):
    r = 0  #设定斯皮尔曼初始值
    good=Y.sum()   #好客户的人数
    bad=Y.count()-good   #坏客户的人数
    #分箱的核心是用机器来选最优的分箱节点
    while np.abs(r) < 1:   #while ,不满足条件时，跳出循环
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        #注意这里是pd.qcut, Bucket：将 X 分为 n 段，n由斯皮尔曼系数决定
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)    # 以斯皮尔曼系数作为分箱终止条件
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X    #  min 就是分箱的节点
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()   #返回 iv
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)    # 返回 d

    pd.set_option('display.max_columns', None)  # 显示完整的列
    pd.set_option('display.max_rows', None)  # 显示完整的行
    print("\033[7;37;41m\t 分箱情况为： \033[0m")
    print("=" * 60)
    print(d4)

    woe=list(d4['woe'].round(3))             #返回 woe
    cut=[]    #  cut 存放箱段节点
    cut.append(float('-inf'))  # 在列表前加 -inf
    for i in range(1,n+1):            # n 是前面分箱的分割数  ，所以分成n+1份
         qua=X.quantile(i/(n+1))     #quantile 分为数 得到分箱的节点
         cut.append(round(qua,4))   # 保留4位小数，返回cut
    cut.append(float('inf')) # 在列表后加inf
    return d4,iv,cut,woe

#对于上述分箱方法不能合理拆分的特征，采用无监督分箱的手动分箱
#贷款以及信用卡可用额度与总额度比例
dfx1, ivx1, cutx1, woex1 = mono_bin(trainDf.SeriousDlqin2yrs, trainDf.RevolvingUtilizationOfUnsecuredLines, n = 10)
dfx2, ivx2, cutx2, woex2 = mono_bin(trainDf.SeriousDlqin2yrs, trainDf.age, n=10)   # 年龄
dfx4, ivx4, cutx4, woex4 = mono_bin(trainDf.SeriousDlqin2yrs, trainDf.DebtRatio, n=20)   #负债比率
dfx5, ivx5, cutx5, woex5 = mono_bin(trainDf.SeriousDlqin2yrs, trainDf.MonthlyIncome, n=10)  #月收入对

pinf = float('inf')          #正无穷大
ninf = float('-inf')         #负无穷大
cutx3 = [ninf, 0, 1, 3, 5, pinf]
cutx6 = [ninf, 1, 2, 3, 5, 7, 9, pinf]   #加了 7，9
cutx7 = [ninf, 0, 1, 3, 5, pinf]
cutx8 = [ninf, 0,1,2, 3, pinf]
cutx9 = [ninf, 0, 1, 3, pinf]
cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]

def self_bin(Y, X, cat):            #自定义人工分箱函数
    good = Y.sum()   #好用户数量
    bad = Y.count()-good   # 坏用户数量
    d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.cut(X, cat)}) #建立个数据框 X-- 各个特征变量 ， Y--用户好坏标签 ， Bucket--各个分箱
    d2 = d1.groupby('Bucket', as_index = True)     #  按分箱分组聚合 ，并且设为 Index
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])  #  添加 min 列 ,不用管里面的 d2.X.min()
    d3['min'] = d2.min().X    #求每个箱段内 X 比如家庭人数的最小值
    d3['max'] = d2.max().X    #求每个箱段内 X 比如家庭人数的最大值
    d3['sum'] = d2.sum().Y    #求每个箱段内 Y 好客户的个数
    d3['total'] = d2.count().Y  #求每个箱段内  总共客户数
    d3['rate'] = d2.mean().Y    # 好客户率
    #WOE的全称是“Weight of Evidence”，即证据权重。WOE是对原始自变量的一种编码形式。是为了 计算 iv 准备的
    #要对一个变量进行WOE编码，需要首先把这个变量进行分组处理（也叫离散化、分箱等等，说的都是一个意思）
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good     # 每个箱段内好用户占总好用户数的比率
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad  # 每个箱段内坏用户占总坏用户数的比率
    #IV的全称是Information Value，中文意思是信息价值，或者信息量。通俗的说就是变量的预测能力
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min'))   #数据框的按min升序排列
    woe = list(d4['woe'].round(3))
    return d4, iv, woe

#对他们就行分箱处理：
dfx3, ivx3, woex3 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
dfx6, ivx6, woex6 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberOfOpenCreditLinesAndLoans'], cutx6)
dfx7, ivx7, woex7 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberOfTimes90DaysLate'], cutx7)
dfx8, ivx8, woex8 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberRealEstateLoansOrLines'], cutx8)
dfx9, ivx9, woex9 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
dfx10, ivx10, woex10 = self_bin(trainDf.SeriousDlqin2yrs, trainDf['NumberOfDependents'], cutx10)

#相关性分析：
import seaborn as sns
corr = trainDf.corr()   #计算各变量的相关性系数
xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']     #x轴标签
yticks = list(corr.index)     #y轴标签
fig = plt.figure(figsize = (10, 8))
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})    #绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
plt.savefig('矩阵热力图.png', dpi=200)
plt.show()

#IV值筛选：通过IV值判断变量预测能力的标准是:
#小于 0.02: unpredictive；0.02 to 0.1: weak；0.1 to 0.3: medium； 0.3 to 0.5: strong
ivlist = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10]      #各变量IV
index=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']    #x轴的标签
fig1 = plt.figure(1, figsize=(8, 5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, ivlist, width=.4)
#ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图
#ax1.bar(x,ivlist,width=.04)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('IV', fontsize=16)   #IV(Information Value),
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()

#建立模型之前，我们需要将筛选后的变量转换为WoE值，便于信用评分
#替换成woe函数
def trans_woe(var, var_name, woe, cut):
    woe_name = var_name+'_woe'
    for i in range(len(woe)):
# len(woe) 得到woe里 有多少个数值
        if i == 0:
            var.loc[(var[var_name] <= cut[i+1]), woe_name] = woe[i]
#将woe的值按 cut分箱的下节点，顺序赋值给var的woe_name 列 ，分箱的第一段
        elif (i > 0) and (i <= len(woe)-2):
            var.loc[((var[var_name] > cut[i]) & (var[var_name] <= cut[i+1])), woe_name] = woe[i]
# 中间的分箱区间，数手指头就很清楚了
        else:
            var.loc[(var[var_name] > cut[len(woe)-1]), woe_name] = woe[len(woe)-1]
#大于最后一个分箱区间的上限值，最后一个值是正无穷
    return var
x1_name = 'RevolvingUtilizationOfUnsecuredLines'
x2_name = 'age'
x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
x7_name = 'NumberOfTimes90DaysLate'
x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'

trainDf = trans_woe(trainDf,x1_name,woex1,cutx1)
trainDf = trans_woe(trainDf,x2_name,woex2,cutx2)
trainDf = trans_woe(trainDf,x3_name,woex3,cutx3)
trainDf = trans_woe(trainDf,x7_name,woex7,cutx7)
trainDf = trans_woe(trainDf,x9_name,woex9,cutx9)

Y = trainDf['SeriousDlqin2yrs']   #因变量
#自变量，剔除对因变量影响不明显的变量
X = trainDf.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                'NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
X = trainDf.iloc[:,-5:]
X.head()

#回归
import statsmodels.api as sm
X1 = sm.add_constant(X)
logit = sm.Logit(Y, X1)
result = logit.fit()

print("\033[7;37;41m\t 展示回归各变量的值： \033[0m")
print(result.summary())

#模型评估：我们需要验证一下模型的预测能力如何。我们使用在建模开始阶段预留的test数据进行检验。
#通过ROC曲线和AUC来评估模型的拟合能力。
#在Python中，可以利用sklearn.metrics，它能方便比较两个分类器，自动计算ROC和AUC
testDf = trans_woe(testDf, x1_name, woex1, cutx1)
testDf = trans_woe(testDf, x2_name, woex2, cutx2)
testDf = trans_woe(testDf, x3_name, woex3, cutx3)
testDf = trans_woe(testDf, x7_name, woex7, cutx7)
testDf = trans_woe(testDf, x9_name, woex9, cutx9)
#构建测试集的特征和标签
test_X = testDf.iloc[:, -5:]   #测试数据特征
test_Y = testDf.iloc[:, 0]     #测试数据标签

#评估
from sklearn import metrics
X3 = sm.add_constant(test_X)
resu = result.predict(X3)    #进行预测

fpr, tpr, threshold=metrics.roc_curve(test_Y, resu)   #评估算法
rocauc = metrics.auc(fpr, tpr)   #计算AUC

print("\033[7;37;41m\t 用预留出的测试集测试模型： \033[0m")
print("\033[7;37;41m\t 模型的AUG的值为： \033[0m")
print('AUG = ',rocauc)


plt.figure(figsize = (8, 5))  #只能在这里面设置
plt.plot(fpr, tpr, 'b',label = 'AUC=%0.2f'% rocauc)
plt.legend(loc = 'lower right', fontsize=14)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylabel('TPR-真正率', fontsize=16)
plt.xlabel('FPR-假正率', fontsize=16)
plt.show()

#设定几个评分卡参数：基础分值、PDO（比率翻倍的分值）和好坏比。
#这里， 我们取600分为基础分值b，取20为PDO （每高20分好坏比翻一倍），好坏比O取20。
p = 20/np.log(2)               #比例因子
q = 600-20*np.log(20)/np.log(2)            #等于offset,偏移量
x_coe = [2.6084, 0.6327, 0.5151, 0.5520, 0.5747, 0.4074]        #回归系数
baseScore = round(q+p*x_coe[0],0)
#个人总评分=基础分+各部分得分
def get_score(coe,woe,factor):
    scores = []
    for w in woe:
        score = round(coe*w*factor, 0)
        scores.append(score)
    return scores
#每一项得分
x1_score = get_score(x_coe[1], woex1, p)
x2_score = get_score(x_coe[2], woex2, p)
x3_score = get_score(x_coe[3], woex3, p)
x7_score = get_score(x_coe[4], woex7, p)
x9_score = get_score(x_coe[5], woex9, p)

def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list
test1 = pd.read_csv(r"C:\Users\Suface\Desktop\Python_code\Give_me_some_credit\Data\cs-test.csv")
test1['BaseScore']=np.zeros(len(test1))+baseScore
test1['x1'] =compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1_score)
test1['x2'] = compute_score(test1['age'], cutx2, x2_score)
test1['x3'] = compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3_score)
test1['x7'] = compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7_score)
test1['x9'] = compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'],cutx9,x9_score)
test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] +test1['x9']  + baseScore

scoretable1 = test1.iloc[:, [1, -7, -6, -5, -4, -3, -2, -1]]  #选取需要的列，就是评分列
scoretable1.head()
scoretable1.to_csv('ScoreData简化版.csv')      #属性名以Xn代替的
print("\033[7;37;41m\t 利用评分卡，在测试集上进行评分(仅展示前10个)： \033[0m")
print(scoretable1[:10])


colNameDict = {'x1': 'RevolvingUtilizationOfUnsecuredLines', 'x2': 'age', 'x3': 'NumberOfTime30-59DaysPastDueNotWorse',
               'x7': 'NumberOfTimes90DaysLate', 'x9': 'NumberOfTime60-89DaysPastDueNotWorse'}
scoretable2 = scoretable1.rename(columns=colNameDict, inplace=False)
scoretable2.to_csv('ScoreData.csv')






print('Done!')
