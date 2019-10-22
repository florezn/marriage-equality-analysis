
## DiD with state-specific linear time trends analysis

In this section I perform a two-way fixed effects with state-specific linear trends analysis for all data and court-decision data only.


```python
%load_ext rpy2.ipython
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

## Loading finalized datasets
data18 = pd.read_csv('weighted_suicides.csv')
court18 = pd.read_csv('court_data.csv')
```

#### DiD with state-specific linear time trends analysis set up

The following sets up a difference-in-difference analysis with state-specific linear time trends for all data and court-decision only data. This notebook expands on notebook 3: [DiD analysis](https://github.com/florezn/marriage-equality-analysis/blob/master/DiD_Analysis.ipynb) so that the main differences are the  `get_lin_trends()` function as well as the option to add linear trends in `did_model()`. `get_lin_trends()` creates state-specific linear time trends by multiplying state dummy variables with a continuous `Year` vector that contains values 1 to the length of time period (15 or 16 depending on whether lag of log unemployment variable is included in the model), inclusive. `lin_trends` option added to `did_model()` adds the linear time trends to the explanatory matrix when `lin_trends` argument is specified as `True`. The same procedure remains as in notebook 3 such that the analysis constitutes 7 different functions and 4 tables summarizing results.

Results for all data with unweighted observations, all data with weighted observations, court-only data with unweighted observations, and court-only data with weighted observations are outputted in Tables V - VIII, respectively.



```python
def get_lin_trends(data):
    
    State_dum = pd.get_dummies(data['State'])
    
    ##Creating state-specific linear time trends matrix
    Years = data.Year - (min(data.Year) - 1)
    lin_trends = (State_dum.transpose() * Years.values).transpose()
    
    #Setting up column names so that state dummies and linear trends variables
    #do not have the same column name
    l = []
    for state in data.State.unique():
        l.append(state + "_lintrend")
                 
    lin_trends.columns = l
    
    return  lin_trends
```


```python
def did_model(data, regressors, lagged=False, lin_trends=False, weighted=False):
    X = data[regressors]
    State_dum = pd.get_dummies(data['State'])
    Year_dum = pd.get_dummies(data['Year'])
    
    if weighted:
        y = data.weighted_lnrate
    else:
        y = data.lnrate
        
    if lagged:
        Year_dum = Year_dum.drop(1999, axis = 1)
        X = X.dropna(axis = 0, how = 'any')
        y = y[data.Year != 1999]
        
    if lin_trends and lagged:
        trends = get_lin_trends(data[data.Year != 1999])
        X = X.join([Year_dum, State_dum, trends])
    elif lin_trends:
        trends = get_lin_trends(data)
        X = X.join([Year_dum, State_dum, trends])
    else:            
        X = X.join([Year_dum, State_dum])
                           
    return sm.OLS(y, X)

def cluster_robust_se(data, model, lagged=False):
    lag = int(lagged)
        
    cluster = np.array(range(len(data.State.unique())) * (len(data.Year.unique()) - lag))
    cluster.sort()
    
    cov_clus = sm.stats.sandwich_covariance.cov_cluster(model, cluster, use_correction = True)
    se_clus_robust = sm.stats.sandwich_covariance.se_cov(cov_clus)
   
    return se_clus_robust

def p_val_clus_robust(model, se_clus_robust):
    p = stats.t.sf(np.abs(model.params / se_clus_robust), model.df_resid - 1) * 2
    
    return p

##Creating coefficient table with cluster robust standard errors and p-values for treatment variable by model
def model_table_results(model, se, p):
    d = {'Coefficient': pd.Series(model.params.index),
         'Estimate': pd.Series(model.params.values),
         'Cluster_Robust SE': pd.Series(se),
         'p-value': pd.Series(p)}
    
    return pd.DataFrame(d)[['Coefficient', 'Estimate', 'Cluster_Robust SE', 'p-value']]
```


```python
def run(data, regressors, lagged=False, lin_trends=False):
    
    model = did_model(data, regressors, lagged, lin_trends).fit()
    wmodel = did_model(data, regressors, lagged, lin_trends, weighted = True).fit()
    
    model_clus_se = cluster_robust_se(data, model, lagged)
    wmodel_clus_se = cluster_robust_se(data, wmodel, lagged)
    
    p = p_val_clus_robust(model, model_clus_se)
    wp = p_val_clus_robust(wmodel, wmodel_clus_se)
    
    table = model_table_results(model, model_clus_se, p)
    wtable = model_table_results(wmodel, wmodel_clus_se, wp)
    
    return table, wtable
```


```python
reg_lin0, wreg_lin0 = run(data18, ['Treatment'], lin_trends = True)
reg_lin1, wreg_lin1 = run(data18, ['Treatment', 'lnunemp'], lin_trends = True)
reg_lin2, wreg_lin2 = run(data18, ['Treatment', 'lnunemp', 'lnunemp_lag'], lagged = True, lin_trends = True)
reg_lin3, wreg_lin3 = run(data18, ['Treatment', 'lnunemp', 'lninc_cap'], lin_trends = True)
reg_lin4, wreg_lin4 = run(data18, ['Treatment', 'lnunemp', 'lninc_cap', 'lnunemp_lag'], lagged = True, lin_trends = True)
```


```python
court_lin0, wcourt_lin0 = run(court18, ['Treatment'], lin_trends = True)
court_lin1, wcourt_lin1 = run(court18, ['Treatment', 'lnunemp'], lin_trends = True)
court_lin2, wcourt_lin2 = run(court18, ['Treatment', 'lnunemp', 'lnunemp_lag'], lagged = True, lin_trends = True)
court_lin3, wcourt_lin3 = run(court18, ['Treatment', 'lnunemp', 'lninc_cap'], lin_trends = True)
court_lin4, wcourt_lin4 = run(court18, ['Treatment', 'lnunemp', 'lninc_cap', 'lnunemp_lag'], lagged = True, lin_trends = True)
```


```python
def aggregate_models(model_list):
    
    model_index = ['Model 0', 'Model 1', 'Model 2', 'Model 3', 'Model 4']
    
    estimates = []
    std_errors = []
    p_vals = []

    for model in model_list:
        estimates.append(model.Estimate[0])
        std_errors.append(model['Cluster_Robust SE'][0])
        p_vals.append(model['p-value'][0])
    
    d = {'Treatment Coeff Estimate': pd.Series(estimates, index = model_index),
         'Cluster_Robust SE': pd.Series(std_errors, index = model_index),
         'p-value': pd.Series(p_vals, index = model_index)}
    
    return pd.DataFrame(d)[['Treatment Coeff Estimate', 'Cluster_Robust SE', 'p-value']]
```


```python
#Table V: Unweighted observations with all data
aggregate_models([reg_lin0, reg_lin1, reg_lin2, reg_lin3, reg_lin4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Coeff Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.095085</td>
      <td>0.045111</td>
      <td>0.035412</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.098096</td>
      <td>0.044701</td>
      <td>0.028537</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.071302</td>
      <td>0.042618</td>
      <td>0.094811</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.097881</td>
      <td>0.044638</td>
      <td>0.028662</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.071265</td>
      <td>0.042661</td>
      <td>0.095313</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table VI: Weighted observations with all data
aggregate_models([wreg_lin0, wreg_lin1, wreg_lin2, wreg_lin3, wreg_lin4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Coeff Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.002452</td>
      <td>0.001203</td>
      <td>0.041934</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.002547</td>
      <td>0.001244</td>
      <td>0.041044</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.001832</td>
      <td>0.001342</td>
      <td>0.172715</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.002543</td>
      <td>0.001242</td>
      <td>0.040904</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.001830</td>
      <td>0.001341</td>
      <td>0.173044</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table VII: Unweighted observations with court data
aggregate_models([court_lin0, court_lin1, court_lin2, court_lin3, court_lin4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Coeff Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.199562</td>
      <td>0.061402</td>
      <td>0.001266</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.214050</td>
      <td>0.049197</td>
      <td>0.000018</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.178943</td>
      <td>0.049255</td>
      <td>0.000326</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.225976</td>
      <td>0.052837</td>
      <td>0.000025</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.189325</td>
      <td>0.052864</td>
      <td>0.000395</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table VIII: Weighted observations with court data
aggregate_models([wcourt_lin0, wcourt_lin1, wcourt_lin2, wcourt_lin3, wcourt_lin4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Coeff Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.005280</td>
      <td>0.001401</td>
      <td>0.000192</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.005650</td>
      <td>0.001450</td>
      <td>0.000117</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.004965</td>
      <td>0.001399</td>
      <td>0.000444</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.005702</td>
      <td>0.001431</td>
      <td>0.000082</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.005016</td>
      <td>0.001382</td>
      <td>0.000331</td>
    </tr>
  </tbody>
</table>
</div>


