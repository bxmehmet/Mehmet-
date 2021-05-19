"""    BU  KISIM  A ŞIKKININ CEVABINI İÇERİR MEHMET KORU 201722161033 1.ÖĞRETİM """
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



vido=pd.read_csv("dataset/vgsales.csv")

vido[["Rank","Year","NA_Sales","Global_Sales","JP_Sales","EU_Sales","Other_Sales"]]=vido[["Rank","Year","NA_Sales","Global_Sales","JP_Sales","EU_Sales","Other_Sales"]].interpolate(method="linear").astype(float) ##### eksik verileri kendisi doldurdu
vido=pd.get_dummies(vido,columns=["Genre"],drop_first=True)
print(vido.head())
print(vido.columns)

y= vido["Genre_Strategy"].astype(float) 
X=vido[["NA_Sales","EU_Sales","Global_Sales","Other_Sales","JP_Sales","Year"]] 
print(y)
print(X)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)  

scaler=MinMaxScaler()  
X_train_sc=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)     
X_test_sc=pd.DataFrame(scaler.fit_transform(X_test),columns=X_test.columns)        
print(X_train)
print(X_train_sc)

lr=LinearRegression()
model=lr.fit(X_train_sc,y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred=lr.predict(X_test_sc)
MSE=mean_squared_error(y_true=y_test,y_pred=y_pred)
MAE=mean_absolute_error(y_true=y_test,y_pred=y_pred)

print("MSE: Mean_squared_eror {}".format(MSE)) 
RMSE=sqrt(MSE)
print("MAE : Mean_absolute_error {}".format(MAE)) 
print("RMSE  {}".format(RMSE)) 
print(len(y_pred))

y_pred=(y_pred>0.05)
accuracy=metrics.accuracy_score(y_pred=y_pred,y_true=y_test)
print("Model accuracy {} ".format(accuracy))

df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred}) 
print(df)

print("Confusuion matrisi\n\n {}".format(confusion_matrix(y_pred=y_pred,y_true=y_test)))    

print(classification_report(y_pred=y_pred,y_true=y_test))



y_pred_train=model.predict(X_train_sc).astype("int64")
y_pred_test=model.predict(X_test_sc).astype("int64")
y_pred_test=(y_pred_test>0.51)

accuracy_train=accuracy_score(y_pred=y_pred_train,y_true=y_train)
accuracy_test=accuracy_score(y_pred=y_pred_test,y_true=y_test)
y_pred_test=(y_pred_test>0.05)

print("Eğitim Accuracy {}".format(accuracy_train))
print("Eğitim test {}".format(accuracy_test))


plt.figure(figsize=(10,8))
plt.bar(X_train.columns.tolist(),lr.coef_[0])
plt.xticks(rotation=50, size=5)
plt.show()

cm1=cm(y_true=y_test,y_pred=y_pred_test)
sns.heatmap(cm1,annot=True,fmt=".0f")
plt.xlabel("predict Values")
plt.ylabel("Actual Values")
plt.title("Accuracy {:.2f}".format(accuracy_test),size=10)
plt.show()

"""

Veri setimizi Başta okuduk 
okuduktan sonra interpolate() fonksiyonu bize linear bir şekilde  eksik  kayıp verilerimizi kendisi doldurdu.
y bağımlı değişkenimizi "Genre_Strategy" kolon nuna atama yapıyoruz.Dummies ile elde ettiğimiz.
X bağımsız değişkenimiz verisetimizde obeject ve içeriği uygun olmayan kolonları atıyoruz .
verilerimizi train test split olarak ayırıyoruzz ayırıyoruz,%30 nu alıyoruz.
verilerimizi MinMaxScaler transform aracılığyla normalize ediyoruz. (X_train,X_test leri )
Multi LinearRegression bir bağımlı ona karşılık bir dizın bağımsız değişkenleri arasında analizi yapar. bizde 1 bağımlı 6 bağımsız üzerinde çalışıyoruz.
modelimizi fit ettik uygun hale getiridk
verilerimizin katsayılarına ,coef, intercept e baktık 
MSEortalama kare hatasıdır regresyon eğrisinin bir dizi noktaya ne kadar yakın olduğunu söyler 0 a yaklaşması iyidir.
MAE mutlak hatadır  iki sürekli değişken arasındaki farkın ölçüsüdür.
RMSE tahminleyicinin tahmin ettiği değerler ile gerçek değerleri arasındaki uzaklığın bulunmasında sıklıkla kullanılan bir ölçüttür .
accuracy modelin doğruluğudur kalitesini tespit etmede çok önemli bir kriterdir
gerçek değer ve tahmin değerleri karşılaştırıyoruz
hata matrisini çıkarıyoruz hata matrisinde 9 yanlışa 222 yanlış olmuş tahmin 9 yanlış
precision recall f1 score support tablosunu çıkardık. Precision doğru olarak ne kadar tahmin edildiğinin bir ölçüsüdür. Mümkün olduğu kadar yüksek olmalıdır.
recalll gerçek pozitif değerlerin oranıdır, f1 score sıflandırıcının ne kadar iyi performans gösterdiğinin bir ölçüsüdür. Bu ölçüler veri analizinde çok önem arz eder
Son olarak ta Eğitim accuracy Eğitim tesleri elde ettik ben bu verileri kullanırken y_pred i filtreledim çünkü hata veriyordu 
Eğitim accuracy  hakkında örnek verecek olursak : Oyun satışları  dükkan dan satışı  biraz düşük performans gösterirken iken, E-ticaret İnternet üzerinden  biraz daha performans göstermektedir.
Verimde bağımlı değişkeni seçmek zor olduğundan,Orjinal verimi BOZMADAN,herhangi bir ekleme çıkarma yapmadan sadece kod kullanarak dummies değişkeni ile seçtim. 


"""

