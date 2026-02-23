result = adfuller(df['Close'])
print("ADF:", result[0], "| p-value:", result[1])

df_diff = df['Close'].diff().dropna()

result_diff = adfuller(df_diff)
print("After differencing → p-value:", result_diff[1])