import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold,cross_val_score

# Reading the train dataset
df_train = pd.read_csv('MobileTrain.csv')

# Reading the test dataset
df_test = pd.read_csv('MobileTest.csv')

#Ranking dataset according to price range

df_train["rank_by_price"]=df_train["price_range"].rank()
df1=df_train
df1

#Sorting above dataset according to ranked_price_range

df1.sort_values(by=["rank_by_price"])

#Ranking on all the features using rank()

RankedData = df1.rank()
RankedData.sort_values(by="price_range")

#Ranking all the features separately to correct output
#Because not all features are good when values are high or low
#It depends on each and every feature.
r= df1
r["rank_by_price"] = r["price_range"].rank()
r["rank_by_battery"] = r["battery_power"].rank(ascending=False)
r["rank_by_blueooth"] = r["blue"].rank(ascending=False)
r["rank_by_clockspeed"] = r["clock_speed"].rank(ascending=False)
r["rank_by_DualSIM"] = r["dual_sim"].rank(ascending=False)
r["rank_by_fc"] = r["fc"].rank(ascending=False)
r["rank_by_4G"] = r["four_g"].rank(ascending=False)
r["rank_by_InternalMemory"] = r["int_memory"].rank(ascending=False)
r["rank_by_mdep"] = r["m_dep"].rank(ascending=False)
r["rank_by_weight"] = r["mobile_wt"].rank(ascending=True)
r["rank_by_ncores"] = r["n_cores"].rank(ascending=False)
r["rank_by_pc"] = r["pc"].rank(ascending=False)
r["rank_by_height"] = r["px_height"].rank(ascending=False)
r["rank_by_width"] = r["px_width"].rank(ascending=False)
r["rank_by_ram"] = r["ram"].rank(ascending=False)
r["rank_by_sch"] = r["sc_h"].rank(ascending=False)
r["rank_by_scw"] = r["sc_w"].rank(ascending=False)
r["rank_by_talktime"] = r["talk_time"].rank(ascending=False)
r["rank_by_3G"] = r["three_g"].rank(ascending=False)
r["rank_by_touchscreen"] = r["touch_screen"].rank(ascending=False)
r["rank_by_wifi"] = r["wifi"].rank(ascending=False)
r.head()

data=pd.read_csv('MobileTrain.csv')
data.head()

#splitting independent and dependent features
x= data.drop('price_range',axis=1)
y=data['price_range']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
logit_model= LogisticRegression()
logit_model.fit(x_train, y_train)
y_pred = logit_model.predict(x_test)


#feature importance and ranking
coefficients = logit_model.coef_

avg_importance =np.mean(np.abs(coefficients),axis=0)
log_feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': avg_importance})
log_feature_importance = log_feature_importance.sort_values('Importance', ascending=True)
log_feature_importance .sort_values(by=['Importance'],ascending=False,inplace=True)
log_feature_importance['rank']=log_feature_importance['Importance'].rank(ascending=False)
log_feature_importance

import pickle
#for model
pickle.dump(logit_model,open('rank_model.pkl','wb'))

#for_scaler
pickle.dump(scaler,open('scaling_features.pkl','wb'))
