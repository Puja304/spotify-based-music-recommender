import pandas as pd

#load file
weather = pd.read_csv("vancouverWeather.csv", header=0, parse_dates=['datetime'])

#print(weather.columns)

#keep only the columns we think are relevant
weather = weather[['datetime', 'temp','precip','windspeed','cloudcover','icon']]

#look at each feature and see what needs to be changed
print(weather.dtypes)

#just in case there's an issue with fields being na
weather = weather.dropna()

#need to convert icon into a different type
encoded = pd.get_dummies(weather['icon'], prefix='icon')
weather.drop(columns='icon')
weather = pd.concat([weather, encoded], axis=1)

weather.to_csv("weather-cleaned.csv", index=False)