############ BU KISIM B NİN CEVABIDIR MEHMET KORU 201722161033

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import  math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

f=pd.read_csv("dataset/vgsales.csv")
f[["Rank","Year","NA_Sales","Global_Sales","JP_Sales","EU_Sales","Other_Sales"]]=f[["Rank","Year","NA_Sales","Global_Sales","JP_Sales","EU_Sales","Other_Sales"]].interpolate(method="linear").astype("int32")
print(f.columns)
print(f.info())
print(f.columns)
print()
f=pd.get_dummies(f,columns=["Publisher"],drop_first=True)
print(f.columns)


y=f["Global_Sales"]

X=f[["Rank","Other_Sales",'Publisher_bitComposer Games',"EU_Sales","JP_Sales",'Publisher_mixi, Inc','Publisher_id Software','Year','Publisher_Zushi Games']]
# X=f[["Rank","JP_Sales",'Publisher_bitComposer Games',"EU_Sales","Other_Sales",'Publisher_mixi, Inc','Publisher_id Software','Year']]
# X=f[["Rank","JP_Sales",'Publisher_bitComposer Games',"EU_Sales","Other_Sales",'Publisher_mixi, Inc']]
# X=f[["Rank","JP_Sales","EU_Sales","Other_Sales",'Publisher_mixi, Inc','Year']]
# X=f[["Rank","JP_Sales","EU_Sales","Other_Sales",'Year']]

######### bağımsız değişkenlerde 5 tanesi direkt olarak kendi verimin Publisher olanlar kendi dummiesler


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)


scaler=MinMaxScaler()  
X_train_sc=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)     
X_test_sc=pd.DataFrame(scaler.fit_transform(X_test),columns=X_test.columns)        
print(X_train)
print(X_train_sc)

lr=LinearRegression()
model=lr.fit(X_train_sc,y_train)
y_pred=lr.predict(X_test_sc)

MSE=mean_squared_error(y_true=y_test,y_pred=y_pred)
print("TEST HATASI MSE {}".format(MSE)) 
RMSE=sqrt(MSE)
print("TEST RMSE :{}".format(RMSE)) 

y_pred_train=lr.predict(X_train_sc)
MSE=mean_squared_error(y_pred=y_pred_train,y_true=y_train)
RMSE=sqrt(MSE)
print("EĞİTİM HATASI : {}".format(MSE))
print("EĞİTİM RMSE {}".format(RMSE))
print("R KARE : {}".format(model.score(X_train_sc,y_train)))


X_multi=sm.tools.tools.add_constant(X_train,prepend=True,has_constant="skip")
mod=sm.OLS(y_train,X_multi)
res=mod.fit()
print(res.summary())

"""
test hataları eğitime sokmadığımız hatalardır, Eğitim hataları her zaman  test hatalarından az olur R kare,yumuşatılmış R kare ise bağımlı değişkenin bağımsız değişkeni bağımsız değişkenin  bağımlı 
değişkeni ne kadar açıkladığını


***OLS REGRASYON :Sıradan En Küçük Kareler Yöntemi, bu yöntem ölçüm sonucu elde edilmiş veri noktalarına mümkün olduğu kadar yakın  geçecek bir işlev eğrisi bulmaya yarar.**

t :Regresyon katsayılarının anlamlılığına ilişkin t istatistiği. t değerleri regresyon katsayılarının standart hatalara bölünmesi ile bulunur.
p: Regresyon katsayılarının anlamlılığına ilişkin p olasılığı.  p<.05 koşulunu sağlayan p değerlerine sahip katsayıların modele katkısının anlamlı olduğu söylenebilir. p nin 0.05 üstü atılır 
s: Regresyon denkleminin standart hatası.
R : muliple regrasyonda bağımlı ile bağımsız değişkenin arasındaki ilişkiyi verir	
R**2 :Çoklu açıklayıcılık katsayısı.Çoklu korelasyon katsayısının karesidir. Bağımsız değişkenlerin bağımlı değişkeni ne oranda açıkladığını gösterir.


istatiksel çoklu regrasyon analizi yapılırken 3 adım uygulanabilir bunlar; 1. ileriye doğru seçme (forward),2.Adım Regresyon (Stepwise) , 3.Geriye Doğru Çıkarma (Backward Elemination)
Bu üç yöntemden eğer diğer yöntemlerin kullanılması için önemli gerekçeler yoksa “Standart Çoklu Regresyon” un kullanılılabilirr. 
biz  p değerlrine göre p>[t] > 0.05 ise bunları sırayla verimizden çıkarıcağız ve Regrasyon analizini daha uygun hale getireceğiz  null(0)hipotez testinin red etmiş oluyoruz 
R**2 ayrıca bağımsız değişkenin bağımlı değişkeni nasıl açıkladğını bize söyler

1. atacağımız değer 0.990 olan Publisher_Zushi Games  
2. atacağımız değer  0. 979 olan  Publisher_id Software  
3.atacağımız değer 0.970  Publisher_bitComposer Games   
4.atacağımız değer 0.880   Publisher_mixi, Inc 

teker teker atarak R kareyi etkisini göreceğiz, yukarıda gösterdim (40-41-42-43-44 satırda teker teker çıkardım. )

!!!Atmadan önce  R squared 0.797, adj.r squared 0.797!!!

1. değer atılınca Publisher_Zushi Games R**2:  0.797 , adj R-squared : 0.797  demekki bu verinin bize bir katkısı yok 
2. değer atılınca Publisher_id Software R**2:  0.797 ,adj R-squared: 0.797  bağımlı  değişkene bir etkisi yok
3. değer atılınca Publisher_bitComposer Games R**2: 0.797 ,adj R-squared 0.797 bağımlı değişkene bir etkisi yok
4. değer atılınca  Publisher_mixii Inc R**2 :   0.797 ,adj R-squared 0.797 bağımlı değişkene bir etkisi yok atılabilir
böylelikle  bağımlı değişkene etkisi olmayan fazlalık oluşturan verileri çıkardık 


coef:negatif olanlar bize yararı yoktur (y değerine bakılmalı yinede) geriye doğru azalmaya sebeb olacak en etkili değişkenimiz EU_Sales (avrupa satışları)
en etkisiz veri ise (coef) Rank -4x10^-5 
const u prepend=False vereek atabiliriz fakat coef yüksek p<0.05 den küçük o yüzden atmadım 

skewnes 14.75  için veri çok yüksektir tamamen sağa çarpıktır  +1 den yükseklerde sağa çarpılır - olursa sola çarpılır
kurtosis 550 çok sivridir  k>3 sivrilmeye başlar k<3 için basıktır







"""



