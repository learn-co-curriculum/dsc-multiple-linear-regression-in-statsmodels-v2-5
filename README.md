
# Multiple Linear Regression in Statsmodels

## Introduction

In this lecture, you'll learn how to run your first multiple linear regression model.

## Objectives
You will be able to:
* Introduce Statsmodels for multiple regression
* Present alternatives for running regression in Scikit Learn

# Statsmodels for multiple linear regression

This lecture will be more of a code-along, where we will walk through a multiple linear regression model using both Statsmodels and Scikit-Learn. 

Remember that we introduced single linear regression before, which is known as ordinary least squares. It determines a line of best fit by minimizing the sum of squares of the errors between the models predictions and the actual data. In algebra and statistics classes, this is often limited to the simple 2 variable case of $y=mx+b$, but this process can be generalized to use multiple predictive variables.

## Auto-mpg data

The code below reiterates the steps we've taken before: we've created dummies for our categorical variables and have log-transformed some of our continuous predictors. 


```python
import pandas as pd
import numpy as np
data = pd.read_csv("auto-mpg.csv") 
data['horsepower'].astype(str).astype(int)

acc = data["acceleration"]
logdisp = np.log(data["displacement"])
loghorse = np.log(data["horsepower"])
logweight= np.log(data["weight"])

scaled_acc = (acc-min(acc))/(max(acc)-min(acc))	
scaled_disp = (logdisp-np.mean(logdisp))/np.sqrt(np.var(logdisp))
scaled_horse = (loghorse-np.mean(loghorse))/(max(loghorse)-min(loghorse))
scaled_weight= (logweight-np.mean(logweight))/np.sqrt(np.var(logweight))

data_fin = pd.DataFrame([])
data_fin["acc"]= scaled_acc
data_fin["disp"]= scaled_disp
data_fin["horse"] = scaled_horse
data_fin["weight"] = scaled_weight
cyl_dummies = pd.get_dummies(data["cylinders"], prefix="cyl")
yr_dummies = pd.get_dummies(data["model year"], prefix="yr")
orig_dummies = pd.get_dummies(data["origin"], prefix="orig")
mpg = data["mpg"]
data_fin = pd.concat([mpg, data_fin, cyl_dummies, yr_dummies, orig_dummies], axis=1)
```


```python
data_fin.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 392 entries, 0 to 391
    Data columns (total 26 columns):
    mpg       392 non-null float64
    acc       392 non-null float64
    disp      392 non-null float64
    horse     392 non-null float64
    weight    392 non-null float64
    cyl_3     392 non-null uint8
    cyl_4     392 non-null uint8
    cyl_5     392 non-null uint8
    cyl_6     392 non-null uint8
    cyl_8     392 non-null uint8
    yr_70     392 non-null uint8
    yr_71     392 non-null uint8
    yr_72     392 non-null uint8
    yr_73     392 non-null uint8
    yr_74     392 non-null uint8
    yr_75     392 non-null uint8
    yr_76     392 non-null uint8
    yr_77     392 non-null uint8
    yr_78     392 non-null uint8
    yr_79     392 non-null uint8
    yr_80     392 non-null uint8
    yr_81     392 non-null uint8
    yr_82     392 non-null uint8
    orig_1    392 non-null uint8
    orig_2    392 non-null uint8
    orig_3    392 non-null uint8
    dtypes: float64(5), uint8(21)
    memory usage: 23.4 KB


This was the data we had until now. As we want to focus on model interpretation and still don't want to have a massive model for now, let's only inlude "acc", "horse" and the three "orig" categories in our final data.


```python
data_ols = pd.concat([mpg, scaled_acc, scaled_weight, orig_dummies], axis= 1)
data_ols.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>acceleration</th>
      <th>weight</th>
      <th>orig_1</th>
      <th>orig_2</th>
      <th>orig_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>0.238095</td>
      <td>0.720986</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>0.208333</td>
      <td>0.908047</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>0.178571</td>
      <td>0.651205</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>0.238095</td>
      <td>0.648095</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>0.148810</td>
      <td>0.664652</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## A linear model using Statsmodels

Now, let's use the statsmodels.api to run our ols on all our data. Just like for linear regression with a single predictor, you can use the formula $y \sim X$, where, with $n$ predictors, X is represented as $x_1+\ldots+x_n$.




```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
```


```python
formula = "mpg ~ acceleration+weight+orig_1+orig_2+orig_3"
model = ols(formula= formula, data=data_ols).fit()
```

Having to type out all the predictors isn't practical when you have many. Another better way than to type them all out is to seperate out the outcome variable "mpg" out of your data frame, and use the a `"+".join()` command on the predictors, as done below:


```python
outcome = 'mpg'
predictors = data_ols.drop('mpg', axis=1)
pred_sum = "+".join(predictors.columns)
formula = outcome + "~" + pred_sum
```


```python
model = ols(formula= formula, data=data_ols).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.726</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.723</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   256.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 08 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.86e-107</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:09</td>     <th>  Log-Likelihood:    </th> <td> -1107.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2224.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   387</td>      <th>  BIC:               </th> <td>   2244.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   16.1041</td> <td>    0.509</td> <td>   31.636</td> <td> 0.000</td> <td>   15.103</td> <td>   17.105</td>
</tr>
<tr>
  <th>acceleration</th> <td>    5.0494</td> <td>    1.389</td> <td>    3.634</td> <td> 0.000</td> <td>    2.318</td> <td>    7.781</td>
</tr>
<tr>
  <th>weight</th>       <td>   -5.8764</td> <td>    0.282</td> <td>  -20.831</td> <td> 0.000</td> <td>   -6.431</td> <td>   -5.322</td>
</tr>
<tr>
  <th>orig_1</th>       <td>    4.6566</td> <td>    0.363</td> <td>   12.839</td> <td> 0.000</td> <td>    3.944</td> <td>    5.370</td>
</tr>
<tr>
  <th>orig_2</th>       <td>    5.0690</td> <td>    0.454</td> <td>   11.176</td> <td> 0.000</td> <td>    4.177</td> <td>    5.961</td>
</tr>
<tr>
  <th>orig_3</th>       <td>    6.3785</td> <td>    0.430</td> <td>   14.829</td> <td> 0.000</td> <td>    5.533</td> <td>    7.224</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>37.427</td> <th>  Durbin-Watson:     </th> <td>   0.840</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  55.989</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.648</td> <th>  Prob(JB):          </th> <td>6.95e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.322</td> <th>  Cond. No.          </th> <td>2.18e+15</td>
</tr>
</table>



Or even easier, simply use the `.OLS`-method from statsmodels.api. The advantage is that you don't have to create the summation string. Important to note, however, is that the intercept term is not included by default, so you have to make sure you manipulate your `predictors` dataframe so it includes a constant term. You can do this using `.add_constant`.


```python
import statsmodels.api as sm
predictors_int = sm.add_constant(predictors)
model = sm.OLS(data['mpg'],predictors_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.726</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.723</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   256.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 08 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.86e-107</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:09</td>     <th>  Log-Likelihood:    </th> <td> -1107.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2224.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   387</td>      <th>  BIC:               </th> <td>   2244.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>   16.1041</td> <td>    0.509</td> <td>   31.636</td> <td> 0.000</td> <td>   15.103</td> <td>   17.105</td>
</tr>
<tr>
  <th>acceleration</th> <td>    5.0494</td> <td>    1.389</td> <td>    3.634</td> <td> 0.000</td> <td>    2.318</td> <td>    7.781</td>
</tr>
<tr>
  <th>weight</th>       <td>   -5.8764</td> <td>    0.282</td> <td>  -20.831</td> <td> 0.000</td> <td>   -6.431</td> <td>   -5.322</td>
</tr>
<tr>
  <th>orig_1</th>       <td>    4.6566</td> <td>    0.363</td> <td>   12.839</td> <td> 0.000</td> <td>    3.944</td> <td>    5.370</td>
</tr>
<tr>
  <th>orig_2</th>       <td>    5.0690</td> <td>    0.454</td> <td>   11.176</td> <td> 0.000</td> <td>    4.177</td> <td>    5.961</td>
</tr>
<tr>
  <th>orig_3</th>       <td>    6.3785</td> <td>    0.430</td> <td>   14.829</td> <td> 0.000</td> <td>    5.533</td> <td>    7.224</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>37.427</td> <th>  Durbin-Watson:     </th> <td>   0.840</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  55.989</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.648</td> <th>  Prob(JB):          </th> <td>6.95e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.322</td> <th>  Cond. No.          </th> <td>2.18e+15</td>
</tr>
</table>



## Interpretation

Just like for single multiple regression, the coefficients for our model should be interpreted as "how does Y change for each additional unit X"? Do note that the fact that we transformed X, interpretation can sometimes require a little more attention. In fact, as the model is built on the transformed X, the actual relationship is "how does Y change for each additional unit X'", where X' is the (log- and min-max, standardized,...) transformed data matrix.

## Linear regression using scikit learn

You can also repeat this process using Scikit-Learn. The code to do this can be found below. The Scikit-learn is generally known for its machine learning functionalities and generally very popular when it comes to building a clear data science workflow. It is also commonly used by data scientists for regression. The disadvantage of scikit learn compared to Statsmodels is that it doesn't have some statistical metrics like the p-values of the parameter estimates readily available. For a more ad-hoc comparison of Scikit-learn and statsmodels, you can read this blogpost: https://blog.thedataincubator.com/2017/11/scikit-learn-vs-statsmodels/.


```python
from sklearn.linear_model import LinearRegression
```


```python
y = data_ols['mpg']
linreg = LinearRegression()
linreg.fit(predictors, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# coefficients
linreg.coef_
```




    array([ 5.04941007, -5.87640551, -0.71140721, -0.29903267,  1.01043987])



The intercept of the model is stored in the `.intercept_`-attribute.


```python
# intercept
linreg.intercept_
```




    21.472164286075383



## Why are the coefficients different in scikit learn vs Statsmodels?

You might have noticed that running our regression in Scikit-learn and Statsmodels returned (partially) different parameter estimates. Let's put them side to side:


|  | Statsmodels | Scikit-learn  |
|-------|------|------|
|  intercept| 16.1041|21.4722|
| acceleration | 5.0494| 5.0494|
| weight | -5.8764|-5.8764|
| orig_1 | 4.6566|-0.7114|
| orig_2 | 5.0690|-0.2990|
| orig_3 | 6.3785|	1.0104|

These models return equivalent results! 
We'll use an example to illustrate this. Remember that minmax-scaling was used on acceleration, and standardization on log(weight). 

Let's assume a particular observation with a value of 0.5 for both acceleration and weight after transformation, and let's assume that the origin of the car = `orig_3`. The predicted value for mpg for this particular value will then be equal to:
- 16.1041 + 5.0494 \* 0.5+ (-5.8764) \* 0.5 + 6.3785 = 22.0691 according to the Statsmodels 
- 21.4722 + 5.0494 \* 0.5+ (-5.8764) \* 0.5 + 1.0104 = 22.0691 according to the Scikit-learn model

The eventual result is the same. The extimates for the categorical variables are the same "up to a constant", the difference between the categorical variables, in this case 5.3681, is added in the intercept!

You can make sure to get the same result in both Statsmodels and Scikit-learn, by dropping out one of the `orig_`-levels. This way, you're essentially forcing the coefficient of this level to be equal to zero, and the intercepts and the other coefficients will be the same. 

This is how you do it in Scikit-learn:


```python
predictors = predictors.drop("orig_3",axis=1)
```


```python
linreg.fit(predictors, y)
linreg.coef_
```




    array([ 5.04941007, -5.87640551, -1.72184708, -1.30947254])




```python
linreg.intercept_
```




    22.482604160455665



And Statsmodels:


```python
pred_sum = "+".join(predictors.columns)
formula = outcome + "~" + pred_sum
model = ols(formula= formula, data=data_ols).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.726</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.723</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   256.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 08 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.86e-107</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:09</td>     <th>  Log-Likelihood:    </th> <td> -1107.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2224.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   387</td>      <th>  BIC:               </th> <td>   2244.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   22.4826</td> <td>    0.789</td> <td>   28.504</td> <td> 0.000</td> <td>   20.932</td> <td>   24.033</td>
</tr>
<tr>
  <th>acceleration</th> <td>    5.0494</td> <td>    1.389</td> <td>    3.634</td> <td> 0.000</td> <td>    2.318</td> <td>    7.781</td>
</tr>
<tr>
  <th>weight</th>       <td>   -5.8764</td> <td>    0.282</td> <td>  -20.831</td> <td> 0.000</td> <td>   -6.431</td> <td>   -5.322</td>
</tr>
<tr>
  <th>orig_1</th>       <td>   -1.7218</td> <td>    0.653</td> <td>   -2.638</td> <td> 0.009</td> <td>   -3.005</td> <td>   -0.438</td>
</tr>
<tr>
  <th>orig_2</th>       <td>   -1.3095</td> <td>    0.688</td> <td>   -1.903</td> <td> 0.058</td> <td>   -2.662</td> <td>    0.043</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>37.427</td> <th>  Durbin-Watson:     </th> <td>   0.840</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  55.989</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.648</td> <th>  Prob(JB):          </th> <td>6.95e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.322</td> <th>  Cond. No.          </th> <td>    9.59</td>
</tr>
</table>



## Summary

Congrats! You now know how to build a linear regression model with multiple predictors in both Scikit-Learn and Statsmodels. Before we discuss the model metrics in detail, let's go ahead and try out this model on the Boston Housing Data Set!
