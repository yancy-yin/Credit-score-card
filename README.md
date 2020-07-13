# Credit-score-card-Give-me-some-credit
Credit score card, using "Give me some credit" dataset, with detailed describe in chinese.
信用评分卡 开发流程

一. 数据获取

面向的客户群体： 
（1）存量客户数据（已开展相关融资类业务的客户，包括个人客户和机构客户）
（2）潜在客户数据（未来拟开展相关融资类业务的客户）
数据集描述： 
（1）名称：Give Me Some Credit （2）来源：Kaggle上一个经典的评分卡案例 （3）内容： 基本属性：包括了借款人当时的年龄 偿债能力：包括了借款人的月收入、负债比率 信用往来：两年内35-59天逾期次数、60-89天逾期次数、90天或高于90天逾期的次数 财产状况：包括了开放式信贷和贷款数量、不动产贷款或额度数量 贷款属性：暂无 其他因素：包括了借款人的家属数量（不包括本人在内） 时间窗口：自变量的观察窗口为过去两年，因变量表现窗口为未来两年 （3）属性： 'ID':'用户ID' 'SeriousDlqin2yrs':'好坏客户' 'RevolvingUtilizationOfUnsecuredLines':'可用额度比值' 'age':'年龄' 'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天笔数' 'DebtRatio':'负债率' 'MonthlyIncome':'月收入' 'NumberOfOpenCreditLinesAndLoans':'信贷数量' 'NumberOfTimes90DaysLate':'逾期90天笔数' 'NumberRealEstateLoansOrLines':'固定资产贷款量' 'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天笔数' 'NumberOfDependents':'家属数量'
样本数量：150000条
特征数量：10个
二. 数据清洗

目的：为了将获取的原始数据转化为可用作模型开发的格式化数据
方法： （1）缺失值处理 直接删除含有缺失值的样本 根据样本之间的相似性填补缺失值 根据变量之间的相关关系填补缺失值 
（2）异常值处理 根据个人经验，如年龄范围、家庭人口数量等 根据直方图观察 根据箱型图观察 （3）重复值删除 （4）不平衡样本处理
意义：保证数据的完整性、全面性、合法性、唯一性
三. 数据分箱

定义：是一种局部平滑方法，通过考察“邻居”（周围的值）来平滑存储数据的值，是对连续变量离散化的一种称呼
目的：去噪，将连续数据离散化，增加粒度
表示：用“箱的深度”表示不同的箱里有相同个数的数据，用“箱的宽度”来表示每个箱值的取值区间
分类： （1）等距分段：是指分段的区间是一致的，比如年龄以10年作为1个分段 
（2）等深分段：是先确定分段数量，然后令每个分段中数据数量大致相等 
（3）最优分段：又称监督离散化，通过将属性值域划分为区间，用来减少给定连续属性值的个数 （4）递归划分：是一种基于条件推断查找较佳分组的算法 （5）人工手动分箱：针对上述方法无法合理拆分的特征，进行手动分箱
意义： （1）离散化后的特征对异常数据有很强的稳定性。如年收入特征>10000是1类，否则是0。如果特征没有离散化，一个异常数据“年收入100000”会给模型造成很大的干扰 （2）单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，增加拟合能力 （3）特征离散化后，模型会更稳定，如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人 （4）特征离散化后，可以简化回归模型，降低模型过拟合的风险 （5）可以将缺失作为独立的一类带入模型
四. 相关性分析

定义：对两个或多个具备相关性的变量元素进行分析，从而衡量两个变量因素的相关密切程度
要求：相关性的元素之间需要存在一定的联系或者概率才可以进行相关性分析
常用的衡量随机变量相关性的方法： （1）Pearson相关系数：即皮尔逊相关系数，用于横向两个连续性随机变量间的相关系数 （2）Spearman相关系数：即斯皮尔曼相关系数，用于衡量分类定序变量间的相关程度 （3）Kendall相关系数：即肯德尔相关系数，也是一种秩相关系数，但它所计算的对象是分类变量
评价手段：相关性矩阵或矩阵热力图 （1）相关系数表示两两相关的程度，所以只有对角矩阵的对角线是变量和自身的相关系数，其值永远是1 （2）矩阵中的小数代表相关性程度，越接近1越相关 （3）通常线性系数大于0.7说明线性相关度比较高 （4）符号为负表示负相关
解决相关性过高/过低的方法： （1）增大样本量 （2）岭回归 （3）逐步回归法 （4）主成分回归 （5）人工去除
意义：当自变量较多时，可以判断有无多重共线性问题
五. 变量选择

定义：通过统计学的方法，筛选出对违约状态影响最显著的指标，主要包括
分类： （1）单变量特征选择方法 （2）基于机器学习模型的方法
评价指标： （1）WOE（Weight of Evidence） 作用： WOE的含义类似于信息熵反映了自变量取值对目标变量的影响 WOE越大表示该特征正向作用越大 它对数据进行了归一化处理，也就是将所有不同特征划在了统一的尺度上 公式：WOE=ln(good attribute/bad attribute) （2）IV（Information Value） 作用： WOE没有考虑分组中样本占整体样本的比例，如果一个分组的WOE值很高，但是样本数占整体样本数很低，则对变量整体预测的能力会下降 IV值考虑了分组中样本占整体样本的比例，相当于WOE的加权求和 计算特征内部的信息含量，如果信息含量足够大，则表示其为有价值的特征 标准： 小于 0.02: 预测能力极低（Unpredictive） 0.02 到 0.1 之间: 预测能力较低（Weak） 0.1 到 0.3 之间: 预测能力中等（Medium） 0.3 到 0.5 之间: 预测能力较强（Strong）
意义： （1）剔除对因变量影响不明显的变量/特征 （2）选取比较重要的变量加入模型，预测强度可以作为判断变量是否重要的一个依据
六. 模型建立与评估

常用模型： （1）逻辑回归（Logistics Regression） 逻辑回归经过信贷历史的反复验证是有效的 模型比较稳定相对成熟 建模过程透明而不是黑箱 不太容易过拟合 （2）梯度提升决策树（Gradient Boosting Decision Tree） （3）极端梯度增强算法（Extreme Gradient Boosting， XGBoost） （4）轻量级梯度提升机（Light Gradient Boosting Machine， Light GBM）
评价指标： （1）ROC（Receiver Operating Characteristic）曲线 定义：画在二维平面上的曲线，平面的横坐标是False Positive Rate(FPR)，纵坐标是True Positive Rate(TPR) 作用：ROC曲线高于对角线的程度越大，则效果越好 （2）AUC（Area Under Curve）值 定义：AUC的值就是处于ROC曲线中，对角线下方部分面积的大小 作用：通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的性能
评估属性： （1）区分能力 （2）预测能力 （3）稳定性
七. 构建评分卡

参数： （1）基础分值 （2）比率翻倍的分值 PDO （3）好坏比
评价指标： （1）ROC曲线 （2）KS（Kolmogorov-Smirnov）曲线
计算方法：个人总评分 = 基础分 + 各部分得分
八. 生成评分卡
