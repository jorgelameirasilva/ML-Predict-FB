import xgboost as xgb
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_df(df, split=False):
    df["target"] = df["Close"].shift(-1)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.dayofweek
    X = facebook_price[["Date", "Open", "High", "Low", "Close"]]
    y = facebook_price["target"]
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
    return X, y

facebook_price = yf.download("FB", end="2020-03-01" )
X_train, X_test, y_train, y_test = get_df(facebook_price, True)

clf = xgb.XGBRegressor()
clf.fit(X_train, y_train.values)

y_pred = clf.predict(X_test)
plt.plot(y_pred[:100])
plt.plot(list(y_test.values[:100]))
plt.show()

new_prices = yf.download("FB", start ="2020-03-02")
X, y = get_df(new_prices)

y_pred = clf.predict(X)
plt.plot(y_pred[:30])
plt.plot(list(y.values[:30]))

plt.show()

