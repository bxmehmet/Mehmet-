# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 16:02:31 2021

@author: Mehmet-PC
"""
## kütüphane
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import  SVC  
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
%matplotlib inline


"""
BUNLAR 2 Lİ KOMBİNASYONLA BAĞIMSIZ DEĞİŞKENLER

SEPAL -  ÇANAK YAPRAK 
PETAL - TAÇ YAPRAK
LENGTH - UZUNLUK
WİDTH   - GENİŞLİK 

BU 3  BAĞIMSIZ DEĞİŞKENLER 
İRİS SETOSA 
İRİS VİRGİNİCA
İRİS VERİSOLOR


"""

iris=pd.read_excel("iris.xls")
print(iris.head()) # #ilk 5 veriye baktık

grp=iris.groupby('iris').agg(["mean","median"])
print(grp)

"""
4 tane bağımsız değişkenleri  ortalama ve medyan olarak ayırdı hesapladı buna göre
ortalama ile medyan birbine yakınsa veriler hemen hemen simetrik dağılmıştır çok az sayıda aykırı değer (outlier)-
bulunduğunu  gösterir KUTU GRAFİĞİ BOX PLOT plot aykırı değerler için en ,iyi istatiksel araçlardan biridir!

"""
grp1=iris.groupby("iris").std()    ##standart sapma dağılımı verir
print(grp1)
"""
STANDART SAPMA VEYA VARYANS  verilerin ortalama değer civarında ne kadar geniş yayıldığını bir göstergesidir
YORUM: standart sapmalar fazla olmadığı için biraz daha sivri bir grafik beklenir yani merkezden dağılım azdır daha yakındır
standart sapma medyan ortalam aykırı değerler için bakılır
eğer standart sapma fazla olduysa demek ki fazla aykırı değerler olduğu anlaşılır

"""


## KUTU GRAFİĞİ BOX PLOT()
"""
Kutu yada yaaty çizgi grafiği veri setinin özetinin beş ayrı değer ile minumum değer alt çeyreklik(%25 lik dilim) medyan (%50) lik dilim -
üst çeyreklik %75 lik dilim ve maksimum değer gibi özellikleri verir
seaborn kütüphanesi kullanırız
"""
###¹ hazır kodu

sns.set(style="ticks") 
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x='iris',y='sepal length',data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='iris',y='sepal width',data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='iris',y='petal length',data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='iris',y='petal width',data=iris)
plt.show()

## ---------------------------------------------------------------------------------------------------------------------


# KEMAN GRAFİĞİ  HAZIR KODLAR


sns.set(style='whitegrid')
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.violinplot(x='iris',y='sepal length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='iris',y='sepal width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='iris',y='petal length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='iris',y='petal width',data=iris)
plt.show()


"""

Bir keman grafiği, kutu grafiğine benzer bir rol oynar. Verilerin, 
bir (veya daha fazla) kategorik değişkenin (bizim durumumuzda çiçek türleri) dağılımını gösterir,
 böylece bu dağılımlar karşılaştırılabilir. Keman grafiğinin ortasındaki çizgide de, 
 verilerin noktalar halinde nerelerde yoğunlaştığı görülebilmektedir.

"""


print("verimizin boyut :\n",iris.shape)

print(iris.describe())   ##m istatiksel hesapları verir 

print(iris.info())   ### sayısal ,kategorik ifadeleri verir 

print(iris.isnull().sum()) #verilerimizde  değersiz değerler var mı


### seabron kütüphanesi bize tek  kodla ikili bağımsız değikenlerin ilişkilerini verir

print(sns.pairplot(iris))


### sıcaklık grafiği ilişkisi
plt.figure(figsize=(10,11))
sns.heatmap(iris.corr(),annot=True, cmap="coolwarm")
plt.plot()



"""
1 e yakın olan yani koyu kırmızı olanlar iyi korelasyon dur 


"""


plt.figure(figsize=(16,9))
plt.title('İris Çiçeği Türlerine Göre Çanak Yaprak Uzunluğu ve Genişliği Dağılımı')
sns.scatterplot(iris['sepal length'], iris['sepal width'], hue = iris['iris'], s= 100);


plt.figure(figsize=(16,9))
plt.title('İris Çiçeği Türlerine Göre Taç Yaprak Uzunluğu ve Genişliği Dağılımı')
sns.scatterplot(iris['petal length'], iris['petal width'], hue = iris['iris'], s= 100);




sns.pairplot(iris,hue="iris",height=3)
### burdan virginica setosa versicolar çeşitlerinin birbiniden ayırmını rahat b,irşekilde yapabilir

"""

PAİRPLOT YORUMU:
    
1- iris setosa için   özellikli çiçeğin  petal witdh (taç yaprak genişliği ) az olduğu görülmektedir 1 . geçmemektedir
    sepal length yani çanak yaprak uzunluğu 4-6 cm,  sepal width fazla olduğu görülür 
    
2- iris veriscolor için  sepal with çanak yaprak genişliği 1-2 aralağındadır 

3- iris verginica için   sepal length  çanak yaprak uzunluğu 6-8 arasındadır  petal length  6-8 aralığındaıdr
   petal witdh 1.5 -2.5 aralığı


"""
fig, axes = plt.subplots(2, 2, figsize=(16,9))
axes[0,0].set_title("Çanak Yaprak Genişliği Dağılımı")
axes[0,0].hist(iris['sepal width'], bins=5);
axes[0,1].set_title("Çanak Yaprak Uzunluğu Dağılımı")
axes[0,1].hist(iris['sepal length'], bins=7);
axes[1,0].set_title("Taç Yaprak Genişliği Dağılımı")
axes[1,0].hist(iris['petal width'], bins=5);
axes[1,1].set_title("Taç Yaprak Uzunluğu Dağılımı")
axes[1,1].hist(iris['petal length'], bins=6);

### BELLİ ARALIKTA FREKANSLARI VERİR



sns.FacetGrid(iris,hue="iris",height=5).map(sns.distplot,"petal width").add_legend();


sns.FacetGrid(iris,hue="iris",height=5).map(sns.distplot,"petal length").add_legend();


sns.FacetGrid(iris,hue="iris",height=5).map(sns.distplot,"sepal width").add_legend();


#####♣SONUÇLAR
"""
Veri kümesi dengelidir, yani her üç tür için eşit kayıtlar mevcuttur.

Dört sayısal veri barındıran sütunumuz varken, analiz etmeyi hedeflediğimiz veri olan sadece bir kategorik sütunumuz vardır (çiçek türleri).
Taç yaprak genişliği ile taç yaprak uzunluğu arasında güçlü bir korelasyon mevcuttur.
Setosa türleri, küçük boyutlu olmasından dolayı dolayı en kolay ayırt edilebilen türdür.
Versicolor ve Virginica türleri genellikle karıştırılır ve bazen ayrılması zordur. Ancak genellikle Versicolor türünün boyutları daha ortalama değerdedir. Buna karşın virginica türünün boyutları daha büyüktür

"""





###### MAKİNE ÖĞRENMESİ

# =============================================================================
# ŞABLON
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import  SVC  
from sklearn import preprocessing


#  Veri okuma, Bağımlı Bağımsız değişken çekme
veri=pd.read_excel("iris.xls")
print(veri.head()) # ilk 5 veriye baktık

# =============================================================================
# ALGORİTMAYI BELİRLEMEK İÇİN GÖRSELLEŞTİRME KULLANACAĞIZ
#  HAZIR KOD THE İRİS DATASET
# =============================================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # sadece 2 değişken alabildik çünkü max 3 boyut görsel var
y = iris.target    ## labelencoder olarak geldi 0 1 2 


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()   

# =============================================================================
# HAZIR KODLAR THE İRİS DATASET 
# =============================================================================

X=veri.iloc[:,:4].values  # bağımısz  değişken
y=veri.iloc[:,4:].values  # bağımlı değişken


print("corelation matrixi \n",veri.corr()) ## corelation  

## Verilerin eğitim ve tes için bölünmesi

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

# Normalize edip eğitim ve uygulamaya sokacağız

sc=StandardScaler()
X_train=sc.fit_transform(x_train)  # eğit ve uygula
X_test=sc.transform(x_test)   # eğtilenden öğren ve uygula yeniden (eğitimeye gerk yoktur)

# =============================================================================
# SINIFLANDIRMA BAŞLIYOR!
# =============================================================================
"""
  X_train eğitim için kullanıcak  verilerin %33ü test oluşturur 67 si train
  fit etmek onu eğitmek ,transofrm ise kullanmak manasına geliyor
"""



#  SVM 

svc=SVC(kernel="linear")  ## linear doğrusal olarak ayrım noktası bulmaya çalışacak
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print("SVC\n",cm)
#  linear olarak 7 yanlış 1 doğru



#  SVC HATA İLE  SEÇİLEBİLECEK ALGORTİMADIR


###. KNN ALGORİTMASI İYİ SONUÇ VERMEDİ 2 HATA 
# knn=KNeighborsClassifier(n_neighbors=5,metric="manhattan")
# """
# metric bizim mesafeyi ölçme de kullanacağımız matematiksel model dir
# mesela euclidean manhattan chebyshev minkowski vb yöntemler mevcuttur
 
# METRİC VE HATA KARŞILAŞTIRMASI:
#     manhattan :2 hata
#     euclidean: 2 hata
#     “chebyshev” : 2 hata
          
# """
# knn.fit(X_train,y_train)
# y_pred=knn.predict(X_test)

# cm=confusion_matrix(y_test, y_pred)

# print("KNN",cm)
# # n neighbors artması daha fazla veriye bakması demektir ama başarıyla alakası yoktur
# #  n=1 yaptığımız zaman confusion matrixde daha iyi sonuç alırız

# -------------------------------------------------------------------------------------------------------------

# ### LOGİSTİC REGRASYON  3 HATALI 
# logr=LogisticRegression(random_state=0)
# logr.fit(X_train,y_train) # xtrain le y train arasında öğrenme yapar
# y_pred=logr.predict(X_test)   # yukardaki öğrenemden sonra tahmin et
# print("tahminler :\n",y_pred)
# print("gercekler:\n",y_test)

# """
# görüldüğü üzere tahminler tamamen yanlış ama logisticregrasyon çalıştı 
# tahminler: erkek erkek erkek erkek kız   erkek erkek erkek
# gerçek:     kız kız    kız    kız  erkek kız  kız   kız 

# """
# # =============================================================================
# #   CONFUSİON MATRİSİ!
# # =============================================================================
# cm=confusion_matrix(y_test, y_pred)
# print("logistic regrassion  :\n",cm)


# --------------------------------------------------------------------------------------------------------
# ####NAİVE BAYES 2 HATA 
# gnb=GaussianNB()
# gnb.fit(X_train,y_train)

# y_pred=gnb.predict(X_test)
# cm=confusion_matrix(y_test,y_pred)
# print("naive bayesinki",cm)


# =============================================================================
# DECİSİON TREE
# =============================================================================

# dtc=DecisionTreeClassifier(criterion="entropy")
# dtc.fit(X_train,y_train)

# cm=confusion_matrix(y_test,y_pred)
# print("DTC\n ",cm)


# # =============================================================================
# # RondomClassifier
# # =============================================================================

# rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
# rfc.fit(X_train,y_train)
# y_pred=rfc.predict(X_test)
# y_proba=rfc.predict_proba(X_test) # olasılık ihtimalleri TRue false

# cm=confusion_matrix(y_test,y_pred)
# print(" RandomForestClassifier\n",cm)

# print(y_proba)
# #  erkek kadın sa mesela bunların olma ihtimalını veriyor
