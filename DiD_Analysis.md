
## Differences-in-differences (DiD) analysis

In this section I perform a two-way fixed effects analysis for all data and court-decision data only.


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

#### DiD analysis set up

The following sets up a difference-in-difference analysis for all data and court-decision only data. I run four sets of the same five models on 1) all data with unweighted observations, 2) all data with weighted observations, 3) court-only data with unweighted observations, and 4) court-only data with weighted observations. The five models include a) no extra controls, b) log of unemployment as a control, c) log of unemployment and its lag as controls, d) log of unemployment and log of income per capita as controls, e) log of unemployment, its lag and log of income per cap as controls.

The analysis constitutes 6 different functions and 4 tables summarizing results. First, `did_model()` sets up the dependent vector and regressor matrix from the data and adds two-way fixed effects. The `weighted` and `lagged` options depend on i) if model is to include weighted vs. unweighted observations and ii) whether lag of log of unemployment is included in the model. `cluster_robust_se()` clusters robust stadard errors by state. Note that cluster robust standard errors are smaller than base standad errors. This is due to the fact that standard errors are anti-correlated within a cluster so that any one observations is likely to deviate less from a random draw of the whole data than from its specific state cluster. `p_val_clus_robust()` calculates the probability of the coefficient estimate value under the null hypothesis (H0: coeff = 0) with cluster robust standard errors. `model_table_results()` creates a coefficient table with cluster robust standard errors and p-values for the treatment variable by model. `run()` calls unweighted/weighted counterparts for each model. `aggregate_models()` aggregates treatment coefficient estimate, cluster robust standard error as well as p-value for associated models.

Results for all data with unweighted observations, all data with weighted observations, court-only data with unweighted observations, and court-only data with weighted observations are outputted in Tables I - IV, respectively.



```python
def did_model(data, regressors, lagged=False, weighted=False):
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
def run(data, regressors, lagged=False):
    
    model = did_model(data, regressors, lagged).fit()
    wmodel = did_model(data, regressors, lagged, weighted = True).fit()
    
    model_clus_se = cluster_robust_se(data, model, lagged)
    wmodel_clus_se = cluster_robust_se(data, wmodel, lagged)
    
    p = p_val_clus_robust(model, model_clus_se)
    wp = p_val_clus_robust(wmodel, wmodel_clus_se)
    
    table = model_table_results(model, model_clus_se, p)
    wtable = model_table_results(wmodel, wmodel_clus_se, wp)
    
    return table, wtable
```


```python
reg0, wreg0 = run(data18, ['Treatment'])
reg1, wreg1 = run(data18, ['Treatment', 'lnunemp'])
reg2, wreg2 = run(data18, ['Treatment', 'lnunemp', 'lnunemp_lag'], lagged = True)
reg3, wreg3 = run(data18, ['Treatment', 'lnunemp', 'lninc_cap'])
reg4, wreg4 = run(data18, ['Treatment', 'lnunemp', 'lninc_cap', 'lnunemp_lag'], lagged = True)
```


```python
court0, wcourt0 = run(court18, ['Treatment'])
court1, wcourt1 = run(court18, ['Treatment', 'lnunemp'])
court2, wcourt2 = run(court18, ['Treatment', 'lnunemp', 'lnunemp_lag'], lagged = True)
court3, wcourt3 = run(court18, ['Treatment', 'lnunemp', 'lninc_cap'])
court4, wcourt4 = run(court18, ['Treatment', 'lnunemp', 'lninc_cap', 'lnunemp_lag'], lagged = True)
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
    
    return pd.DataFrame(d)[['Treatment Estimate', 'Cluster_Robust SE', 'p-value']]

```


```python
#Table I: Unweighted observations with all data
aggregate_models([reg0, reg1, reg2, reg3, reg4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.042321</td>
      <td>0.033953</td>
      <td>0.212986</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.042214</td>
      <td>0.033624</td>
      <td>0.209706</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.044607</td>
      <td>0.034161</td>
      <td>0.192060</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.037793</td>
      <td>0.032027</td>
      <td>0.238372</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.039577</td>
      <td>0.032406</td>
      <td>0.222397</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table II: Weighted observations with all data
aggregate_models([wreg0, wreg1, wreg2, wreg3, wreg4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.002325</td>
      <td>0.000731</td>
      <td>0.001520</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.002307</td>
      <td>0.000722</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.002394</td>
      <td>0.000773</td>
      <td>0.002023</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.002143</td>
      <td>0.000695</td>
      <td>0.002125</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.002232</td>
      <td>0.000747</td>
      <td>0.002892</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table III: Unweighted observations with court data
aggregate_models([court0, court1, court2, court3, court4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.130982</td>
      <td>0.033070</td>
      <td>0.000090</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.123496</td>
      <td>0.034084</td>
      <td>0.000331</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.114384</td>
      <td>0.045779</td>
      <td>0.012931</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.114422</td>
      <td>0.037065</td>
      <td>0.002173</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.102578</td>
      <td>0.045392</td>
      <td>0.024454</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Table IV: Weighted observations with court data
aggregate_models([wcourt0, wcourt1, wcourt2, wcourt3, wcourt4])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment Estimate</th>
      <th>Cluster_Robust SE</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model 0</th>
      <td>-0.003705</td>
      <td>0.000704</td>
      <td>2.423457e-07</td>
    </tr>
    <tr>
      <th>Model 1</th>
      <td>-0.003569</td>
      <td>0.000703</td>
      <td>6.129624e-07</td>
    </tr>
    <tr>
      <th>Model 2</th>
      <td>-0.003551</td>
      <td>0.000956</td>
      <td>2.392365e-04</td>
    </tr>
    <tr>
      <th>Model 3</th>
      <td>-0.003039</td>
      <td>0.000612</td>
      <td>1.027507e-06</td>
    </tr>
    <tr>
      <th>Model 4</th>
      <td>-0.003087</td>
      <td>0.000851</td>
      <td>3.305485e-04</td>
    </tr>
  </tbody>
</table>
</div>


