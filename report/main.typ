#import "ieee.typ": ieee
#import "@preview/frackable:0.2.0": *
#import "@preview/whalogen:0.2.0": ce

#set figure(placement: auto)
#set cite(style: "ieee-3et.al.csl")
#show: ieee.with(
  title: [Evaluation of Machine Learning Approaches to predict Wine Quality],
  abstract: [
    In this paper, four different models, Classification and Regression Tree, Multiple Linear Regression and K-rank with Support Vector Machine were implemented to evaluate and predict wine quality, using physicochemical properties of white and red _vinho verde_ samples. From a 5-fold cross validation, the results showed that is preferable to use Classification and Regression Tree, while Multiple linear regression showed better performance then the baseline. 
  ],
  authors: (
    (
      name: "André Plancha",
      email: "andre.plancha@hotmail.com"
    ),
    (
      name: "Joanna Stark",
      email: "joannaastark@gmail.com"
    ),
  ),
  index-terms: ("Wine quality prediction", "Ordinal Regresison", "CART", "K-Rank"), // TODO
  bibliography: bibliography("refs.bib", style: "ieee"),
  figure-supplement: [Figure],
)
#set math.equation(numbering: none)
#set table.hline(stroke: .4pt)
#set table.vline(stroke: .4pt)
// TODO separate file in different files?

// Abstract
// 

= Introduction

Wine tasting can be referred to as the "critical analysis of wine, either attempting to rank it within some concept of quality, cultivar, or appellation prototype, delineate its sensory diversity, or investigate the origins of its sensory characteristics" @handbook. Wine tasting is a topic of interest to winemakers, sommeliers, researchers, and consumers alike. The top 10 wine producing countries in 2021 produced more than 18 billion tonnes of wine @ourworldindata. Therefore, it is important to access the quality of wine samples without blind tasting every single one, essentially using only sensory attributes.

For that end, in this project we used machine learning techniques to model wine ranking prediction based solely on the physicochemical attributes of wine samples, since wine's "legitimate quality lies in its sensory characteristics derived from its chemistry" @handbook. Using only these characteristics, winemakers can roughly judge their wine samples _en mass_ without a comprehensive quantitative wine assessment for each sample; Consumers can roughly estimate the quality of a sample before buying the sample; and researchers and sommeliers can understand what chemical properties enhance wine quality.


// todo MAKE SECTIONS AS REFERENCES
Section 2 reviews previous research conducted on the wine quality dataset, highlighting key findings and methodologies. Section 3 provides an overview of the dataset, including an analysis of the variables and outliers; the preprocessing of the data is also provided. In Section 4, we outline the methodology used for modeling the data, detailing the algorithms used and why we choose them, and our evaluation metrics.
Section 5 and 6 are reserved for the results and conclusions, respectively, and will include feature importance analysis and possible future research.


= Related Literatures
// psycological research
// wine quality assessment without machine learning techicques
// wine quality assessment with machine learning tecnhiques
// this dataset and other people who used it

In previous research, several attempts have been made to classify wine quality using the same dataset provided by #cite(<dataset_source>, form: "prose"). #cite(<CORTEZ2009547>, form: "prose") concluded that Support Vector Machines (SVM) outperformed Neural Networks (NN) and Multiple Regression (MR) techniques. They note that this might have been due to the fact that NN have a tendency to fall into local minima during training. Subsequent studies @Kumar have shown that deep learning approaches can provide better prediction accuracy compared to SVM.


#cite(<CORTEZ2009547>, form: "prose") identified sulphates as the most important feature for higher quality wine with SVM, followed by alcohol, citric acid, and residual sugar for white wine. For red wine, the key features were pH, sulfur dioxide, and alcohol. We expect to conclude the same thing.

In another study @Nandan multiple machine learning models including Random Forest, SVM, and Logistic Regression were employed to classify wine quality into three categories: Best, Good, and Poor. Their results indicated that the Random Forest classifier outperformed other models, achieving an accuracy of 98.11%.

// TODO describe diferences between them and ours
// TODO write what we expect based on these related literatures, and with that the impact
= Dataset // bad title

The dataset comprises two separate collections: one for white wine samples containing 4898 records, and another for red wine with 1599 samples ($approx 3:1$ ratio), all from the demarcated region of _vinho verde_, "a unique product from the Minho (northwest) region of Portugal" @CORTEZ2009547. The data was collected from May/2004 to February/2007 and was recorded by a computerized system (_iLab_). Only the most common physicochemical tests were selected to avoid discarding wine examples, which are presented in table @summary. The quality parameter was assessed through blind tastings conducted by at least three evaluators, who rated the wine samples on a scale from 0 to 10. The final score was determined by calculating the median of these evaluations @CORTEZ2009547. The datasets were retrieved from UC Irvine Machine Learning Repository, available at @dataset_source.



#figure(image("variables.png"), caption: "Histograms and boxplots of variables in dataset", scope: "parent", placement: auto)<hists>
In @hists we can visualize variables available in the data using histograms and boxplots after combining the red and white wine datasets.
It is possible to observe that most features show a positive skewness in their distribution, including a lot of outliers; additionally, it is also possible to visualize that about #frackable(3,4) of the wines are white. Looking at the target variable `quality`, we can see that most ratings fall between a 6 or 5, with rare instances of them being outside of these 2; we can also note that there are no wine samples rated 10, or 0 to 2.

== Outliers

Interquartile range (IQR) is a measure of the spread of the data. It defines the difference between the 75th and 25th percentile of the data. Observations that have a value 1.5 times greater than the IQR or a value 1.5 times less than the IQR are declare an outlier, where IQR = Q3 − Q1. The IQR method can be used when variables are not assumed to be normally distributed. @Han This is visualized as a barplot in @hists.

There are some models more sensitive to outliers, like Support Vector Machines (SVMs), and some that are more robust, like Decision Trees. Since we used some sensitive and some robust models, we can take outliers into account when analyzing the results.

#figure(image("summary_table_large_cells.png"), caption: "Physiochemical data statistics of wine", scope: "column", kind: table)<summary>


== Correlation
#figure(image("correlations.png"), caption: "Correlations between features and target", scope: "column")<corrs>
In @corrs we can visualize the linear correlations of each feature with each other, and the linear correlation of each feature with the target variable, using Pearson's correlation coefficient $rho$: the value ranges from -1 to 1, and the greater the absolute value they have, the stronger the linear correlation.

#show "SO_2": ce("SO2")
From the Figure, we can determine that there are a lot of features that are either slightly or very correlated with each other, especially `total sulfur dioxide` (total SO_2) and `free sulfur dioxide` (free SO_2). SO_2 is a compound used by winemakers to "help keep their wine protected from the negative effects of oxygen exposure as well as spoilage microorganisms" @so2. Free SO_2 represents the amount of unbound SO_2 in the sample, or in other words, the amount that the sample can be protected against oxidation and spoilage. This amount is directly correlated to total SO_2, so it is natural that their $rho$ is elevated.
The variables that appear to influence wine quality the most are alcohol, volatile acidity, and density, though the correlations are not particularly strong. Higher alcohol levels are generally associated with better wine quality ratings, while both density and volatile acidity tend to be linked with lower quality ratings. // TODO talk about other variable and feature + target

// TODO talk about correlation between red whine and the other variables

// == Red and White Wine Differences
// TODO

== Preprocessing
// Since we have highly correlated features, we are going to apply Principal Component Analysis (PCA) between our continuous features. PCA is a linear dimensionality reduction that transforms the data onto a new coordinate system using the covariance between the variables, in hopes of reducing the correlation between variables. In other words, the technique transforms correlated variables into uncorrelated principal components. 

// TODO how to do PCA + pseudocode

// To assess the effectiveness of PCA in reducing dimensionality and improving model performance, we are going to evaluate our models with and without doing PCA. This can be observed in @results. // TODO don't like the word "observed" here.

For preprocessing we normalized our continuous features to ensure that our methods do not favor any feature due to its larger or smaller scale. This was especially important for our non-tree based methods.
= Methodology

//[OBSOBS]
//There are many possible ways to analyze and classify wine quality.  [källa]


== Pipeline
A visualization of our pipeline can be found on @pipeline. It consists of a 5-fold cross validation (without repetition) with a normalization of the continuous variables for each fold. We make the normalization within each fold instead of before splitting to not introduce "future information into the training explanatory variables" @norm. After normalizing, we train the model using the train portion and apply it to the test portion. Following this, we create the metrics from the results of the test dataset. Finally, we aggregate by averaging the metrics from every fold.

#figure(image("pipeline.drawio.png"), caption: "Pipeline")<pipeline>

This process is repeated for every model we developed, and the results are compared in Section V. // TODO do reference instead?
The code is available at #link("https://github.com/notPlancha/wine-quality") .

== Algorithms <algos>
Since this problem is of ordinal regression/classification, in theory, regular regression/classification algorithms alone would be able to compute a result, but the results may not be optimal, "because of the information distortion from scale type" @induction_ordinal_trees: doing regular classification would ignore the order of the classes, and doing continuous regression implies quantitative semantics @induction_ordinal_trees. Even when evaluation ratings, those quantitative semantics are being implied when they should not, since, contrary to quantitative variables, the "difference between values cannot be translated to difference in the underlying property" @ordinal_CV, even if they are numbers. In other words, our models should have ordinal invariance, and not "assume predefined intervals between classes" @cem. For example, it is hard to say that the experience for the wine taster is the same when they're tasting 2 wine samples of rating 5, and 1 wine sample of rating 10.

For this project, we compare regular classification/regression models with models adjusted for ordinal regression. For this purpose, we implemented:

#let krank = [_K_-rank method]

- A classification/regression decision tree, using the CART algorithm, introduced by #cite(<cart>, form: "prose") @class_trees_ordinal;
//- An Ordinal Decision Tree (ODT), introduced in @induction_ordinal_trees;
- The #krank, with help of a linear SVM
- Multiple linear regression with gradient descent 

\
=== The CART algorithm

CART is a variation of a Decision Tree (DT) and it can handle both classification and regression tasks. It's one of the most popular implementation of DTs.

The training set is initially split into two subsets using a single feature k and a threshold $t_k$. The values of k and $t_k$ are chosen to maximize the "purity" of the resulting subsets, meaning they should create the most homogeneous groups in terms of class labels. The process then recursively splits each subset, continuing this until a stopping criterion is met. The cost function to be minimized at each step is given by:
$ J(k,t_k) = m_("left")/m "MSE"_("left") + m_("right")/m "MSE"_("right") $

where $m_("right")$ and $m_("left")$ are the number of instances in the left and right subsets, respectively, and $"MSE"_("right")$ and $"MSE"_("left")$ measure the Mean Squared Error (MSE) of these subsets. The MSE for any given node is defined as 

$ "MSE"_"node" = sum _{i in "node"} 
(hat(y)_"node" - y^((i)))^2 $

where $ hat(y)_"node" = 1/m_"node" sum _{i in "node"} y^((i)) $ 

DTs are prone to overfitting since the tree structure adapts itself to the training data, fitting it very closely. To mitigate overfitting, regularization techniques can be applied, such as limiting the maximum depth of the tree. The algorithm will stop growing the tree either when it reaches this maximum depth or when further splitting no longer reduces impurity. Finally, predictions are made by traversing the tree from the root node to a leaf node, based on the features of the input data.

//=== ODT
//An ODT is a DT with focus on ordinal prediction. The approach we're going to follow is the one introduced in #cite(<induction_ordinal_trees>, supplement: "Ch. 4"). The main difference with this algorithm and CART is its splitting strategy.#footnote[More on the 100%]

=== K-rank with SVM
The #krank is "the most popular approach for ordinal regression, in which $K − 1$ [binary] classifiers are trained to rank ordinal categories" @ord2sec. @krank shows a diagram of the process. This method is similar to One-vs-Rest (OvR) methods, but this method takes into account the ordinal nature of the problem, because the model is incorporating multiple consecutive classes. The accompanying model has to have any probability or confidence score output, followed by a transformation to probabilities using, for example, Platt scaling; the SVM model is capable of using the (signed) distance from the separating plane to generating scores @platt.

SVM is a powerful machine learning model, known for its effectiveness in both linear and non-linear classification problems due to the use of kernel functions. This makes it particularly suitable for datasets like the wine quality dataset, where the relationships between features and output labels may not always be linear. The flexibility of SVM, with its ability to employ various kernels, allows for capturing complex patterns that are crucial for accurately predicting wine quality. Additionally, SVM performs well on smaller datasets and can handle high-dimensional feature spaces efficiently, which is relevant for this dataset as it contains multiple physicochemical features. One drawback of SVM is that it can be computationally expensive for very large datasets, but given the size of the wine quality dataset, this is not a significant concern @Han. Another disadvantage of SVM is its sensitivity to class imbalance, which could become an issue in our case, since the ratings are mostly around 5 and 6, as we can observe in @hists.

Our implementation of the SVM is a linear Hard-Margin SVM @svm. It consists of a quadratic optimization problem, with the following structure:

$
  min_(w,b) quad 1/2 ||w||^2 \
  "s.t." quad y_i (w^T x_i + b) >= 1, forall i
$
For the #krank, as explained previously, our model needs to produce probabilities. Therefore we applied a Platt Scaling using the following sigmoid function @platt:

$
  P(y = 1|x) = 1/(1+e^(A f(x) + B)) \
  f(x) = w^T x + b
$
To determine $A$ and $B$, we use the adapted algorithm written by #cite(<note>, form: "prose").

#figure(image("krank.png"), caption: [#krank @simple]) <krank>

=== Multiple linear regression with gradient descent 

Multiple Linear Regression (MLR) models the relationship between multiple independent variables (features) and a dependent variable (target) through a linear combination. The model can be expressed as:

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)
#numbered_eq($hat(y) = bold(theta) dot bold(x)$
)


where $bold(theta) = [theta_0,..., theta_n]$ and $bold(x) = [1, x_1, ..., x_n] $


MLR is employed to predict wine quality based on physicochemical properties. In classical linear regression theory, the optimal parameter vector $hat(theta)$ that minimizes the cost function is determined analytically through the normal equation

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)
#numbered_eq($ hat(theta) = (bold(X)^T bold(X))^(-1) (bold(X)^T bold(Y)) $
)

where X represents the feature matrix and Y the target vector. However, this approach may be susceptible to overfitting and multicollinearity issues, particularly in large-scale datasets. To address these limitations, gradient descent optimization presents an alternative method for model training, iteratively minimizing the Mean Squared Error (MSE) cost function.


The gradient descent algorithm requires computing the partial derivatives of the cost function with respect to each model parameter $theta_j$. The gradient, being a multivariate generalization of the derivative, is calculated as:

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)
#numbered_eq($delta /(delta theta_j) "MSE"(bold(theta)) = 2/m sum_(i=1)^m (bold(theta) bold(x)^((i)) - y^((i)) )^2 bold(x_j)^((i))$
)

This can be expressed more concisely in vector notation as the gradient of the cost function:

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)
#numbered_eq($gradient_theta  "MSE" (bold(theta)) = 2/m  bold(X(X theta - y))$)

This formulation represents batch gradient descent, as it utilizes the complete training dataset at each iteration. The algorithm proceeds by updating the parameter vector in the direction opposite to the gradient, scaled by a learning rate $eta$:

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)
#numbered_eq($bold(theta)^("next step") = bold(theta) - eta gradient_(bold(theta))  "MSE" (bold(theta)) $
)


#let mae = [#emph[MAE#super[#math.mu]]]
#let mamae = [#emph[MAE#super[M]]]
#let cem = $"CEM"^"ORD"$
== Metrics
To evaluate and compare our models, we're going to assess performance metrics. Because our problem is of ordinal nature, our metrics cannot simply be evaluating classification or regression performance, revealing the same issues already discussed in _Algorithms_ (@algos). With that end, we're planning on analyzing 4 metrics:
Mean Absolute Error (#mae), a regression metric;
Accuracy, a classification metric; 
Macro Averaged Mean Absolute Error (#mamae), a metric robust to class imbalance; and Closeness Evaluation Measure (#cem), a metric that focuses on informational closeness. @cem

=== Accuracy
Accuracy is the error rate of the model. In other words, it's the ratio between correct predictions among all predictions. With $n = \#hat(y) = \#y$, 
#set math.equation(numbering: "(1)")
$
  "Accuracy"(y, hat(y)) = 1/(n) sum^(n)_(i = 1)(cases(1 "if" y_i = hat(y)_i, 0 "otherwise"))
$ <acc>
#set math.equation(numbering: none)

The main issue of this metric in this context is that it gives us no information on how close a prediction is to its real value. For example, if a wine sample got a quality of 3, the model predicting a 9 instead of a 4 would result in the same weight for the metric, since they both would result in a 0 in the condition on @acc, breaking Ordinal Monotonicity @cem. Nonetheless, we'll use the metric as a restrictive performance metric, where it's result indicates how exact our model is predicting the qualities.

=== #emph[#mae]
#mae is the "average deviation of the predicted class from the true class" @2009. It's a widely used metric in regression problems, and it's calculated as the average of the absolute differences between predicted and actual values.

$
  mae(y, hat(y)) = 1/(n) sum^(n)_(i = 1)|hat(y)_i - y_i|
$

The metric, as opposed to Accuracy, does somewhat reveal how close a prediction is to a value, and the example noted before would be resolved. With that being said, this metric isn't aware of Ordinal Invariance @cem, discussed previously already. Additionally, the metric does not manage the effect of class imbalance in any way, meaning that a wrongly predicted value in a small class will have the same effect as a class with a lot of records. This is especially problematic in our case since, for example, our model predicting 4 on a 6 is a bigger error predicting a quality of 1 on a 3, not only because 6 is way more common, but also because it can be argued than 1 is closer to 3 than 4 is to 6 when describing ratings in a subjective, intuitive judgment. Nonetheless, the metric will be useful because it's a well known and acceptable metric and it's used a lot, which could make it easier to compare with other non discussed future (or past) models.

=== #emph[#mamae] 
#mamae is a metric derived from #mae, transformed to be more robust to class imbalance, proposed in @2009. Derived from the macro-averaged version of $F_1$, the metric are based on "a sum of the classification errors _across classes_" @2009. With $\#K$ being the number different classes,

$
  mamae(y, hat(y)) = 1/(\#K) sum_(k in K) mae({y | y = k}, {hat(y) | y = k}) \ // these are multisets
  space // This is to push the number to the bottom
$
While this metric resolves the issue of imbalance raised by the other two metrics, it still has the issue of ordinal invariance, since it uses the #mae as a step to the error calculation between true and predicted value.

#let ord = [#text(weight: 600)[`ORD`]]
#let infquant = [$prec.eq$#h(-3pt)$space^b_ord$] // not needed I think
=== #cem
Introduced in @cem, #cem is a metric designed to solve the ordinal invariance problem while still obeying ordinal monotonicity and having into account imbalance. It uses the idea of _informational closeness_: "The more unexpected it is to find an item between $a$ and $b$, the more information such event provides, and the more $a$ and $b$ are informationally closer". With/* $P (x_i infquant a)$ being the probability that a sampled $x_i in x$ is closer to b than a, and */ $"CIQ"^"ORD"$ being the Closeness Information Quantity, and CM being the confusion matrix constructor: // dont like this CM thing but whtv
$
  cem(y,hat(y)) = (sum^n_(i=1)"CIQ"^"ORD" (y_i, hat(y)_i))/(sum^n_(i=1)"CIQ"^"ORD" (y_i, y_i))
$
$
  cem(y,hat(y)) = (sum_(k_1 in K) sum_(k_2 in K)("prox"(k_1, k_2) dot "CM"(k_1, k_2)))/(sum_(k in K) ("prox"(k, k) dot \#{y | y = k}  )) \
  "CM"(k_1, k_2) = \# {(y, hat(y)) | y = k_1 and hat(y) = k_2} \
  "prox"(k_1, k_2) = -log_2((\#{y | y = k_1}/2 + \# {y | k_1 < y <= k_2})/(\#y)) // TODO verify if this is right
$
// @CEMtable smsht smth ,ahtm. // TODO 
/*
#figure(table(
    align: (x, y) => 
      if x == 0 {right}
      else {center},
    rows: 6, columns: 6, stroke: none,
    table.vline(x: 2, start: 2),
    table.hline(y:2, start: 2),
    table.cell(colspan: 6)[True values],table.cell(rowspan: 6)[#rotate(-90deg)[]]
)) <CEMtable>

*/
#cem has been designed to be a metric that follows the 3 properties the authors outline in their article: ordinal invariance, ordinal monotonicity, and imbalance. More information can be found on @cem @other-cem. Because of this, this metric seems more useful for this project; however, since it's a recently developed metric, it has less reliability compared to the others, although it has been used before @roitero. // TODO references should be written as footnotes

/*
$
  "CIQ"^"ORD" (y_i, hat(y_i)) = -log((\#{y | y = y_i}/2 + sum^y_(k=hat(y) + 1) \#{y | y = k}) / 2) \ \
  space
$
*/

= Results and Discussion <results>
To evaluate the models performance, the implemented methods were compared against a basic predictive model. This baseline model predicts the quality in the test set by using the most frequent quality from the training set of each fold, without taking into account any feature; by definition, this coincides with the mode of the column. In most (if not all) folds this will be quality 6. After we have the results of each fold, we average the metrics of each, as explained in Section 4. The results of this model are available on @baseResults; #sym.arrow.t signifies that bigger the metric value, better the prediction, while #sym.arrow.b is bigger the value, bigger the error.

// This is obviously AI but whtv
The results demonstrate differences between the implemented models. The CART algorithm showed the strongest overall performance, achieving a #mae of 0.555 (±0.01), indicating that, on average, its predictions deviate by just over half a point from the true wine quality ratings. This represents a significant improvement over both the baseline (0.642 ±0.02) and the MLR model (0.582 ±0.03). More notably, the #krank underperformed, with a MAE of 0.681 (±0.09), falling below even the baseline performance.

The #mamae metric provides a more balanced view of the model's performance across different quality ratings, accounting for the class imbalance noted in @hists where most wines are rated 5 or 6. The metric, which accounts for class imbalance, reveals a similar pattern with CART achieving the best score of 1.198 (±0.10), followed by MLR at 1.328 (±0.16). The SVM model's #mamae of 1.654 (±0.179) again indicates its struggle with this particular prediction task.

Looking at the #cem metric, which specifically addresses the ordinal nature of wine quality ratings, CART achieved the highest score of 0.570 (±0.01), showing its ability to capture the ordered relationship between quality levels. MLR followed with 0.470 (±0.03), while SVM scored just 0.134 (±0.3), suggesting particular difficulty in handling the ordinal aspect of the prediction task.

All algorithms present small standard deviation, which suggests consistent performance across different cross-validation folds.
#text(size: 12pt)[
#figure(
  table(columns: (1fr, 1fr, 1fr, 1fr, 1fr),stroke: none,
    table.vline(x:1, start:1),
    table.hline(y:1, start:1),
    align: (y,x) => {
      if (y == 0) {right}
      else if (x == 0) {center}
      else {left}
    },
    [],[#mae #sym.arrow.b],[Accuracy #sym.arrow.t],[#mamae #sym.arrow.b],[#cem #sym.arrow.t],
    [Baseline],$0.642 (±0.02)$, $0.433 (±0.02)$, $1.629 (±0.12)$, $0.142 (±0.32)$,
     [MLR],$0.582 (±0.03)$, $0.523 (±0.02)$, $1.328 (±0.16)$, $0.470 (±0.03)$,
     [CART],$bold(0.555) (±0.01)$, $bold(0.556) (±0.01)$, $bold(1.198) (±0.10)$, $bold(0.570 (±0.01))$,
      [K-RANK],$0.681 (±0.09)$, $0.421 (±0.03)$, $1.654 (±0.18)$, $0.134 (±0.30)$
  ), caption: [Results]  
)<baseResults>
]
// == Feature importance
// TODO

// Noticebly AI
The analysis of model performance revealed several important insights about the wine quality prediction task. 

CART showed the best performance across all metrics, and is significantly better than the baseline model, suggesting that wine quality prediction benefits from models capable of capturing non-linear relationships between physicochemical properties. The relatively strong performance of MLR, despite its simplicity, indicates that there are linear relationships between some physicochemical properties and wine quality. However, the gap between MLR and CART's performance suggests that these relationships are not entirely linear, supporting the use of more flexible modeling approaches.

Unexpectedly, the #krank + SVM performed worse than the baseline. This could've indicated that the method was not appropriate to predict ordinal regression, however after further investigation, we concluded that this result is due to the sensitivity that the SVM has towards unbalanced data; when training the multiple models, the #krank divides purposely the data into a very unbalanced matter, specially in classes on the tails, from the transformation into a binary problem#footnote[OvR has the same problem, but it's not common to have so many classes when using OvR methods]. A visual can be seen on @inb. The problem was further inflated because not only was the data already very unbalanced from the start, but also we used a Hard-Margin SVM, which is even more sensible to unbalanced data compared to alternatives this only inflated the problem. This suggests that using the #krank with a SVM is not recommended.

#figure(
  image("inbalance.png", height: 45%), caption: [Demonstration of the imbalance @simple]
)<inb>

The #cem scores for all methods, while showing differences between approaches, remain well below the theoretical maximum of 1. This suggests that alternative approaches specifically designed for ordinal regression could yield better results 

The best accuracy score of 55.6% might reflect the difficulty in predicting wine quality since quality is subjective and predicting the exact rating might be challenging even for human experts. 


= Conclusion
Wine quality assessment has traditionally been dependent on on wine tasting by human experts. Therefore, finding new approaches involving applying machine learning methods to this domain has gained more research attention. While previous research has often approached this as a classification problem by dividing quality into discrete categories, this work aimed to predict quality ratings as a continuous variable using MLR, CART, and SVM with the #krank.

The CART algorithm provided the best performance across all evaluation metrics outperforming both MLR and SVM. While both MLR and CART surpassed the baseline model's performance, the #krank with the SVM underperformed because of class imbalance from the dataset and the nature of the method.

The #cem score for all method leaves room for further improvement, suggesting that alternative approaches specifically designed for ordinal regression could yield better results. Future work might benefit from exploring hybrid methods that better capture both linear and non-linear relationships in the data. Additionally, investigating feature importance and the impact of specific physicochemical properties could provide insights for quality prediction and wine production optimization.

The overall performance levels indicate that while physicochemical properties can predict wine quality reasonable well, there remains uncertainty in the prediction task. These models could potentially serve as support in wine production and reduce the need for extensive taste testing of all samples. The uncertainty of the predictions might also be due to that there is a lot of factors that influence the human perception of wine quality. 

Overall, while there is room for improvement, this work demonstrates that machine learning approaches, particularly CART, can provide valuable tools for supporting wine quality assessment. 

For future work, assessing feature importance, specially from the CART (since it was our best performing model), would give insights on which wine properties are more impactful for its rating; additionally, an interesting study on the differences of the red and white wine could improve models by using knowledge-based rules. Further pre-processing options could also be explored in the future, like Principal Component Analysis or outlier handling, could aid in achieving better scores for future models. Another interesting study would be comparing different tree algorithms with ordinal regression problem, on top of CART. Finally, an analysis on the #krank performance with various models besides Hard-Margin SVM would fill a critical gap in the method originally presented by #cite(<simple>, form: "prose").
// TODO add impact for other types of wine

//Mean Square Error (regression)

//$M = 1/N sum_(i=1)^N (y_i - hat(y))^2$ 