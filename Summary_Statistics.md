
## Summary statistics

This notebook encompasses some statistics about suicide in 2014 as well as summary statistics of the dependent and explanatory variables.


```python
%load_ext rpy2.ipython
import numpy as np
import pandas as pd

## Loading necessary R packages
%R library(ggplot2)
%R library(grid)

## Loading finalized datasets
legal = pd.read_csv('Legalization.csv')
suicides18_all_by_year = pd.read_csv('Suicides_males_18_total_per_year.csv')
suicides_all_by_year = pd.read_csv('Suicides_all_males_total_per_year.csv')

data18 = pd.read_csv('weighted_suicides.csv')
court18 = pd.read_csv('court_data.csv')
```

#### Brief facts about suicide in 2014 and summary statistics of the data

Suicide rates vary across race, gender and age as well as by geographical region and method of suicide. Nationally, older groups consistently have higher suicide rates than younger groups; males outnumber females in lethal attempts; whites outnumber minorities; and in 2014, half of suicides resulted from firearms[^1].The 2014 national suicide rate for the U.S. was 13.0 per 100,000 population[^2], 21.1 per 100,000 population for all males and 3.3 per 100,000 population for males 18 and under[^3]. In total, 1293 men under the age 18 died by suicide in 2014.

**Table I** tabulates the individual states that legalized same-sex marriage before January 2015. I only consider treatment states before 2015 because the Supreme Court legalized marriage equality across the nation mid-way through the year, confounding analysis for 2015.

**Table II** denotes the number of states that legalized same-sex marriage prior to Jan. 2015 by type of law: popular vote, legislature or state court decision. Twenty-four states legalized marriage equality by court decision, eight by legislature and three by popular vote. Recall that court-decision legalization is plausably more exogenous than other means, which leads to a more accurate analysis.

**Figure I** showcases the cumulative number of states that legalized same-sex marriage by the end of each year: 16 by the end of 2013 and 35 by the end of 2014. This aids identification because a greater number of 'treatment' states bolsters precision.

**Table III** displays summary statistics of the control variables for a random sample of the data. Income per capita is in dollars, and the unemployment rate (%) is seasonally adjusted. **Table IV** describes the number of observations and minimum as well as maximum values for each control variable and log of each. Both of these tables ensure reasonable values for the controls so to not have erroneous or outlier data.

The top panel of **Figure II** depicts the national suicide rates for males by year while the lower panel displays the national trends for males 18 and younger. Visiually, the trends differ in both magnitude and shape. Older males (which make up most of the 'all males' category) commit suicide for very different reasons than younger males. Thus, focusing on males 18 and younger reduces noise in the data and targets the most vulnerable population to discriminatory policies.

Lastly, as shown in **Figure III**, the distribution of suicide rates in the data has a long right tale (top panel) so that a log transformation is necessary to attain normality (bottom panel).

[^1]: https://afsp.org/about-suicide/suicide-statistics/
[^2]: http://www.cdc.gov/nchs/products/databriefs/db241.htm
[^3]: see national trends graphs below




```python
## Table I: states that legalized same-sex marriage before January 2015
legal_prior = legal[legal.Year_Passed < 2015]
state_list = pd.DataFrame(legal_prior.values).loc[:, [0, 1, 2, 4]]
state_list.columns = ['State', 'Month', 'Year', 'Method']
state_list.sort_values( by = ['Year', 'Month'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Month</th>
      <th>Year</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Massachusetts</td>
      <td>5</td>
      <td>2004</td>
      <td>court</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Connecticut</td>
      <td>11</td>
      <td>2008</td>
      <td>court</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Iowa</td>
      <td>4</td>
      <td>2009</td>
      <td>court</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Vermont</td>
      <td>9</td>
      <td>2009</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>18</th>
      <td>New Hampshire</td>
      <td>1</td>
      <td>2010</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>21</th>
      <td>New York</td>
      <td>6</td>
      <td>2011</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Maine</td>
      <td>12</td>
      <td>2012</td>
      <td>popvote</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Washington</td>
      <td>12</td>
      <td>2012</td>
      <td>popvote</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Maryland</td>
      <td>1</td>
      <td>2013</td>
      <td>popvote</td>
    </tr>
    <tr>
      <th>2</th>
      <td>California</td>
      <td>6</td>
      <td>2013</td>
      <td>court</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Delaware</td>
      <td>7</td>
      <td>2013</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Minnesota</td>
      <td>8</td>
      <td>2013</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Rhode Island</td>
      <td>8</td>
      <td>2013</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>19</th>
      <td>New Jersey</td>
      <td>10</td>
      <td>2013</td>
      <td>court</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hawaii</td>
      <td>12</td>
      <td>2013</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>20</th>
      <td>New Mexico</td>
      <td>12</td>
      <td>2013</td>
      <td>court</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Oregon</td>
      <td>5</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Pennsylvania</td>
      <td>5</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Illinois</td>
      <td>6</td>
      <td>2014</td>
      <td>legislature</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alaska</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arizona</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Colorado</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Idaho</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Indiana</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Nevada</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>22</th>
      <td>North Carolina</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Oklahoma</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Utah</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Virginia</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>32</th>
      <td>West Virginia</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Wisconsin</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Wyoming</td>
      <td>10</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Kansas</td>
      <td>11</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Montana</td>
      <td>11</td>
      <td>2014</td>
      <td>court</td>
    </tr>
    <tr>
      <th>27</th>
      <td>South Carolina</td>
      <td>11</td>
      <td>2014</td>
      <td>court</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Table II: Number of states that legalized same-sex marriage prior to Jan. 2015 by type of law
pd.DataFrame({'Method': ['Court', 'Legislature', 'Popular Vote'],
              'Number of States': legal_prior.groupby('Method')['State'].count().values})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Number of States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Court</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Legislature</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Popular Vote</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
##Cumulative count of states that legalized same-sex marriage by year, 2003 - 2015

grouped = state_list.groupby('Year')['State']
legal_count_by_year = pd.DataFrame({'Year': grouped.count().index.values, 'Count': grouped.count().values})
state_cumulative = {'Year': np.insert(legal_count_by_year.Year.values, 
                                      [0, 1, 1, 1, 8], [2003, 2005, 2006, 2007, 2015]),
                   'Num_States': np.insert(np.cumsum(legal_count_by_year.Count).values, 
                                           [0, 1, 1, 1, 8], [0, 1, 1, 1, 50])}
state_cumulative = pd.DataFrame(state_cumulative)
```


```r
%%R -i state_cumulative
## Figure I: Cumulative number of states that legalized same-sex marriage by the end of each year

mytheme <- theme_minimal() + theme(plot.title = element_text(size = 15, face = 'bold'), 
                                   axis.title = element_text(size = 15))

ggplot(data = state_cumulative, aes(x = Year, y = Num_States)) + 
geom_point(size = 3, color = '#0072B2') + 
geom_line(color = '#0072B2') +
labs(title = 'Cumulative number of states that legalized \nmarriage equality by the end of each year', 
     y = 'Cumulative number of states') + 
scale_x_continuous(breaks = c(2003:2015)) + mytheme
```


![png](Summary_Statistics_files/Summary_Statistics_6_0.png)



```python
## Table III: Sample of control variables table
controls = data18[['State', 'Year', 'income_cap', 'lninc_cap', 
                       'Unemployment', 'lnunemp', 'unemp_lag', 'lnunemp_lag']]

for col in controls.columns[2:8]:
    controls.loc[ :, col] = controls[col].round(2)


controls.columns = ['State', 'Year', 'Income per Capita', 
                                      'Log of Income per Capita', 
                                      'Unemploymment Rate', 
                                      'Log of Unemployment', 
                                      'Lagged Unemployment Rate', 
                                      'Log of Lagged Unemployment']
```


```python
controls.sample(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Year</th>
      <th>Income per Capita</th>
      <th>Log of Income per Capita</th>
      <th>Unemploymment Rate</th>
      <th>Log of Unemployment</th>
      <th>Lagged Unemployment Rate</th>
      <th>Log of Lagged Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>740</th>
      <td>Washington</td>
      <td>2003</td>
      <td>34663</td>
      <td>10.45</td>
      <td>7.41</td>
      <td>2.00</td>
      <td>7.41</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>295</th>
      <td>Maine</td>
      <td>2006</td>
      <td>34019</td>
      <td>10.43</td>
      <td>4.58</td>
      <td>1.52</td>
      <td>4.88</td>
      <td>1.59</td>
    </tr>
    <tr>
      <th>376</th>
      <td>Mississippi</td>
      <td>2007</td>
      <td>29543</td>
      <td>10.29</td>
      <td>6.09</td>
      <td>1.81</td>
      <td>6.54</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>339</th>
      <td>Michigan</td>
      <td>2002</td>
      <td>30333</td>
      <td>10.32</td>
      <td>6.26</td>
      <td>1.83</td>
      <td>5.19</td>
      <td>1.65</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Maine</td>
      <td>2010</td>
      <td>37102</td>
      <td>10.52</td>
      <td>8.15</td>
      <td>2.10</td>
      <td>8.09</td>
      <td>2.09</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Table IV: Number of observation as well as minimum and maximum values for each control variable

nobs = []
mins = []
maxs = []
means = []
SDs = []

for col in controls.columns[2:8]:
    nobs.append(controls[col].count())
    mins.append(controls[col].min())
    maxs.append(controls[col].max())
    means.append(controls[col].mean())
    SDs.append(controls[col].std())
    


contrls_des = pd.DataFrame([nobs, mins, maxs, means, SDs], 
                           index = ['N', 'Min', 'Max', 'Mean', 'SD'], 
                           columns = ['Income per Capita', 
                                      'Log of Income per Capita', 
                                      'Unemploymment Rate', 
                                      'Log of Unemployment', 
                                      'Lagged Unemployment Rate', 
                                      'Log of Lagged Unemployment'])
contrls_des = contrls_des.transpose()
contrls_des.N = contrls_des.N.astype(int, copy = True)
contrls_des.Min = contrls_des.Min.round(2)
contrls_des.Max = contrls_des.Max.round(2)
contrls_des.Mean = contrls_des.Mean.round(2)
contrls_des.SD = contrls_des.SD.round(2)
```


```python
contrls_des
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>Min</th>
      <th>Max</th>
      <th>Mean</th>
      <th>SD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income per Capita</th>
      <td>800</td>
      <td>20594.00</td>
      <td>64864.00</td>
      <td>36491.78</td>
      <td>7713.97</td>
    </tr>
    <tr>
      <th>Log of Income per Capita</th>
      <td>800</td>
      <td>9.93</td>
      <td>11.08</td>
      <td>10.48</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>Unemploymment Rate</th>
      <td>800</td>
      <td>2.30</td>
      <td>13.66</td>
      <td>5.79</td>
      <td>2.05</td>
    </tr>
    <tr>
      <th>Log of Unemployment</th>
      <td>800</td>
      <td>0.83</td>
      <td>2.61</td>
      <td>1.70</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>Lagged Unemployment Rate</th>
      <td>750</td>
      <td>2.30</td>
      <td>13.66</td>
      <td>5.80</td>
      <td>2.09</td>
    </tr>
    <tr>
      <th>Log of Lagged Unemployment</th>
      <td>750</td>
      <td>0.83</td>
      <td>2.61</td>
      <td>1.70</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```r
%%R -i suicides_all_by_year,suicides18_all_by_year

## Figure II: National suicide rates for males vs. males 18 and younger by year

q1 <- ggplot(suicides_all_by_year, aes(x = Year, y = Crude_Rate)) + 
geom_point(size = 3, color = '#0072B2') + 
geom_line(color = '#0072B2') +
labs( title = 'National trends for all males', y = 'Suicide Rate') + 
theme_minimal() + theme(plot.title = element_text(size = 15, face = 'bold'))

q2 <- ggplot(suicides18_all_by_year, aes(x = Year, y = Crude_Rate)) + 
geom_point(size = 3, color = '#0072B2') + 
geom_line(color = '#0072B2') +
labs(title = 'National trends for males 18 and younger', y = 'Suicide Rate') + 
theme_minimal() + theme(plot.title = element_text(size = 15, face = 'bold'))

grid.newpage()
grid.draw(rbind(ggplotGrob(q1), ggplotGrob(q2), size = "last"))
```


![png](Summary_Statistics_files/Summary_Statistics_11_0.png)



```r
%%R -i data18

## Figure III: Rate of suicide and log rate of suicide distribution plots for males 18 and younger

p1 <- ggplot(data = data18, aes(x = Rate, y = ..density..)) + 
geom_histogram (alpha = .85, color = '#0072B2', fill = "#0072B2", binwidth = .5) + 
labs(title = 'Distribution plots for rate and lnrate of suicide') + 
theme_minimal() + theme(plot.title = element_text(size = 15, face = 'bold'))

p2 <- ggplot(data = data18, aes(x = lnrate, y = ..density..)) + 
geom_histogram (alpha = .85, color = '#0072B2', fill = '#0072B2', binwidth = .1) + theme_minimal()

grid.newpage()
grid.draw(rbind(ggplotGrob(p1), ggplotGrob(p2), size = "last"))
```


![png](Summary_Statistics_files/Summary_Statistics_12_0.png)

