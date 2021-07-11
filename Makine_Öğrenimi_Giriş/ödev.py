
"""
Created on Sat Jul 10 15:51:12 2021
@author: Mehmet-PC
"""
#  ÖDEV 1.0 BENİM YAPTIĞIM MEHMET KORU
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# VERİ YÜKLEME
veri=pd.read_csv("tenis.csv")



 
 ## KATEGORİK > NUMERİK DÖNÜŞÜMÜ
from sklearn  import preprocessing 
veri2=veri.apply(preprocessing.LabelEncoder().fit_transform) ## bütün kolonlar üzerinde labelencoder uygulandı


# outllok kısmı onehot encoder olarak aldık 
c=veri.iloc[:,:1]
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()

#  dataframe leri birleştireceğiz 

outlook=pd.DataFrame(data=c,index=range(14),columns=["o","r","s"])
sonveri=pd.concat([outlook,veri.iloc[:,1:3]],axis=1)
news3=pd.concat([sonveri,veri2.iloc[:,-2:]],axis=1)  # Verimiz tamamen düzeldi



####----------------------------------------------------------------------
## ŞİMDİ VERİLERİ BÖLELİM
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



X=news3.iloc[:,[0,1,2,3,5,6]]
y=news3.iloc[:,[4]]


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

# =============================================================================
# ŞİMDİ P VALUE DEĞERLERE GÖRE GERİ ELEME YAPACAĞIZ VE VERİ SETİMİZİN BAŞARISINI ARTMASINI UMUYORUZ:)
# =============================================================================

model=sm.OLS(y,X).fit()
print(model.summary())  

"""
ols raporuna göre  Play değişkeni çıkartılır
"""


# =============================================================================
# çıkarıldaktan sonra tekrar tahmin edelim
  # Windy verisi çıkarıldı p value > 0.5
# =============================================================================


X=news3.iloc[:,[0,1,2,3,6]]
y=news3.iloc[:,[4]]
x=np.append(arr=np.ones((14,1)).astype(int),values=X,axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred2=lr.predict(x_test)



" y pred2 ile daha düzgün sonuçlar elde edilmiştir."
















