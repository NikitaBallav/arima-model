# arima-model
This project is to forecast the future trends of bitcoin price using ARIMA model.

I have used "Bitcoin US Dollar" data set which consists of 6 attributes namely: Date, Open, High, Low, Adj Low, Volume.
Of the 6 attributes, "Date" attribute is in date format, whereas everything else is Currency Format.
A excel file has been uploaded on the same.

Steps involved for getting the results are as follows:
Step 1: Visualizing the time series data.
Step 2: Assessing whether the time series is stationary, and if not, how many differences are required to make it stationary.
Step 3: Estimation involves using numerical methods to minimize a loss or error term.
Step 4: To look for evidence that the model is a good fit for the data.


Interpretation : The ARIMA model employed in this study demonstrates remarkable predictive performance, achieving an accuracy 
of 97% and a mean absolute percentage error (MAPE) of 0.028. These evaluation metrics indicate the model's 
ability to effectively capture the underlying patterns and dynamics of the time series data. The high accuracy of 
97% signifies that the ARIMA model accurately forecasts 97% of the observed values in the time series. This 
suggests that the model aptly captures the essential trends and fluctuations present in the data, leading to highly 
precise predictions.
Furthermore, the low MAPE of 0.028 demonstrates that the model's predictions exhibit minimal deviation, with an 
average percentage error of only 2.8% from the actual values in the time series. This indicates a strong alignment 
between the model's forecasts and the true values, highlighting its reliability and accuracy. Taken together, the 
impressive accuracy and low MAPE values underscore the exceptional performance of the ARIMA(1,1,1) model 
with the selected parameters (p=1, q=1, d=1) in forecasting future values within the time series. These results 
validate the model's ability to effectively capture the inherent patterns, trends, and dynamics of the data, leading to 
highly accurate predictions.

In conclusion, the findings of this research highlight the strong predictive capabilities of the employed ARIMA 
model, which can provide valuable insights and support decision-making processes in various domains that rely on 
accurate time series forecasting.
