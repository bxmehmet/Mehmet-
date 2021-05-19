# import pandas as pd
# import numpy as np
# from scipy import stats as st
# import matplotlib.pyplot as plt

# ###KULLANACAĞIM VERİ SETİ Video Games Sales data set/vgsales.csv >



# ####TEMEL VERİ İŞLEMLERİ

# f=pd.read_csv("dataset/vgsales.csv",encoding="utf-8",index_col=None,skiprows=0) ### veri setimizi okuyoruz,utf-8 ile dil filtrelemesi yapıyoruz, skiprows = atlama yapar
# print(type(f)) ###pandas DataFrame tipinde
# print(f.head(n=5)) ###n değeri verilmez ise ilk 5 değeri verir
# print(f["Name"].head()) ## Name satırından ilk 5 veri okunur
# print(f.tail(2)) ### veriden sondan 2 si alınır.
# print(f.columns) ### kolon isimleri
# print(f.index) # ##indeks isimleri
# print(f.dtypes) # ###bu metot kolon özellikleri bakılır
# print(len(f))  ### dataset teki kayıtları gösterir
# print(f.head())


# f=f[:-16567] #####veriler çok fazla olduğu için ilk 30 veriyle analize devam edeceğiz
# print(f[:2]) ####sadece ilk 2 satır alır,son indis,son indis dağil değildir
# print(f[::5]) #### 5 atlayarak veri bitimine kadar
# print(f[:10:4]) ##### 0-9 arası 4 atlayarak



# ##NUMPY ANALİZİ

# z=f[["Other_Sales","Global_Sales"]].to_numpy() ###  verilerde str değerleri olduğu için ,İNT VERİLERİ NUMPY A ÇEVİRDİK 
# for i in z.flat:  ###flat fonksiyonu ile satır satır elemanları yazdırdık
#     print(z)

# print(" return numpy değerleri/n tipi{} {}".format(z,type(z)))
# print(z[0:10:3]) ### 0-10  3 atlayarak 10 dağil değil
# z[:3]=50 ## ##0,1,2 50 olarak değiştrildi


# print(z.flatten())  ####tek bir satırda yazar

# print(np.where(z>8,z*2,z)) ### 8 den büyükleri  2 ile çarpıcak   ve yazıcak
# print(" min değeri {}\n max değerler {} ".format(np.argmin(z),np.argmax(z)))
# print("Max değerin satırını döndürür:{}".format(np.argmax(z,axis=1)))
# print("içinde 2 değeri varmı:\t {}".format((z==2).any()))

# print("Medyan:{}\n Mod: {}\n varyans {}\n standart sapma {} ".format(np.median(z),st.mode(z),np.var(z),np.std(z)))
# print("Q1:{}\n Q2 {}\n,Q3 {}".format(np.percentile(z,25),np.percentile(z,50),np.percentile(z,75)))
# ### Q1  VERİLERİN %25İ 1.GRUP ORTASI// Q2 VERİLERRİ %50 YANİ 2 YE BÖLER// Q3  2.GRUP ORTASI VERİLERİN %75 İ
# ### IQR= (Q3-Q1)*1.5 ELDE EDİLİR
# Q1= np.percentile(z,25)
# Q2= np.percentile(z,50)
# Q3=np.percentile(z,75)
# IQR= (Q3-Q1)*1.5
# iç1=(Q3-IQR)
# iç2=(Q1+IQR)
# dış1=(Q3+15)
# dış2=(Q3-15)


# print("İÇ SINIRLARI BULMAK\n {}\n {} ".format((Q3-IQR),(Q1+IQR)))  ##iç sınırları buluruz
# print("DIŞ SINIRLAR: \n{} \n {}".format((Q3+15),(Q3-15)))  ## Dış sınırlar buluruz


# ## 2  Numpay arraylerini karşılaştıralım

# x=f[["NA_Sales","EU_Sales"]].to_numpy()
# z=f[["Other_Sales","Global_Sales"]].to_numpy()

# for i in z.flat,x.flat: 
#     print("Toplam değerleri :{}{}\n Verilerin ortalaması:{} {}\n  verilerin standart sapması {} {} ".format(z.sum(),x.sum(),z.mean(),x.mean(),z.std(),x.std()))





# ###PANDAS DATAFRAME ANALİZİ

# # Temel Veri işlemler


# print(f.count(axis=0)) # sütunları sayıyor 1 olursa satırları sayar
# print(f.T.head())  # Verilerimizin Transpozunu satır sütün yer değişimini yaptık 5 veri için
# print(f[["Name","Year"]].head(n=3)) # name , year satırın altından ilk 3 veri analizi, çif parantez olmalı yoksa eror verir.
# print(f.values[4][-1]) # [4] 4.satırdan alır [-1] sondaki elemanı alır.
# print(f.iloc[0].head(n=4)) # satırları sütün olarak alır ve ilk  4 veri yazılır,aynı işlemi loc da aynı sonuç alınır. iloc >integer location 
# print(f.iloc[0:2]) # verimizden ilk 2 veri çekilir  ilocta son indis alınmaz
# print(f.iloc[[2, 5], [4, 3]]) ##2. ve  5.indis 4. ve 3. kolon alır
# print(f.iloc[2:8:2, 0:8:2]) ##2.indis 7indise dahil 2 şer atlayarak ,0-8 8 dahil değil 2  şer atlar
# print(f.loc[0:4,["Genre","Publisher"]]) ## loc ile son indis dahil olarak çekebilirz
# print(f.loc[5:20, ['Other_Sales',"Global_Sales"]]>2) ## 0 -20 indis belirtilen kolonları 2 den büyükmü kontrolü



# # # # f["Other_Sales"].cat.categories = ["very good", "good", "very bad"]
# # # # print(f["Other_Sales"])


# #####KOLON YENİDEN ADLANDIRMA
# f.rename(columns={'NA_Sales': 'New_Name'}, inplace=True)  ##Sütun ismi değiştirnme işlemi
# print(f)



# #####KOLON ÇIKARMA
# print(f.columns)# şimdi istediğimiz kolonları verimizden çıkaralım
# columns=['Rank','Platform',"New_Name"] 
# f.drop(columns,inplace=True,axis=1)
# print(f)



# # ###2.YOL
# f.drop(["JP_Sales","Other_Sales"],axis="columns",inplace=True) # istediğimiz satırı silebiliriz birden fazla olursa parantez kullanılmalı.
# print(f.head())


# print(f.loc[0:3,["Genre","Publisher"]])  ###istediğimiz verileri loc ile son indisi dahil olarak çekebiliriz iloc kullanılırsa hata verecektir
# f=f[:-1] ### son satırı silme işlemi

# f=pd.read_csv("dataset/vgsales.csv",encoding="utf-8",index_col=None)  ##### verileri eski hale çevirelim
# f=f[:-16547] ### 50 veri ile Analiz yapacağız
# print(f)



# ## VERİ SIRALAMA
 
# print(f.sort_values(by='Other_Sales', ascending=False))   ### Other_Sales  e göre  Büyükten küçüğe sıralama
# print(f.sort_index( ascending=False)) ### indekslere göre büyükten küçüğe sıralama

# # DATAFRAME BİRLEŞTİRME

# df2=f.iloc[2:10,round(len(f.columns)/4)]   ### len(f.columuns) 11 oluyor/4 >2.75 çıkıyor 2.75 i round ile yuvarlıyor 3 çıkıyor  ve bu atlama değeri oluyor yani f.iloc[:,3] oluyor
# df3=f.iloc[10:19,round(len(f.columns)/2):len(f.columns)] ##   len(f.columns)=11  11/3=3.66 round ile 4 yuvarlanacak f.iloc[:,4:11]  aslında
# df4=pd.concat([df2,df3],sort=True) # birleştirme yapar ve sıralamayı kapatır, kesişim kümesini temizler
# DMX=df4[df4["JP_Sales"].isnull()].fillna("ImChampion") # nan olanları str,int olarak doldurabiliriz
# print(DMX)
# ### 2 DataFrame iloc ile Belirli kısımlarını aldık / concat ile birleştridik sort ile sıralama yaptık
# ### iloc[A:B,C:D]  A-B indis numara C-D kolon Filtreler ilocta son indis (exclusive) dahil değildir, Locta inclusive (dahil) dir.

# #### STR VERİLERİNDE HARF DÜZENLEME
# f["Name"]=f["Name"].str.capitalize() #ilk harf büyük gerisi küçük olur
# print(f.head())


# # VERİ FİLTRELEME

# print(f[f["Year"]>2010].head()) # #filtreleme yapıldı 2010 den büyükler verilir ,eğer parantezi çıkarırsak True False döndürecektir.
# q7=f["Year"]>2011
# print(q7.head())
# print(f.loc[lambda f: f["Other_Sales"] >8])    ## filtreleme yapılır
# q7=f[(f["Year"]>2009)&(f["Year"]<2015)] ###2009 dan  büyük 2015 den küçük filtreleme yaptık
# print(q7.head())
# q7=f[f["Genre"]=="Misc"].sort_index() ### Genre de Misc olanları indeks numaralarına göre sıralama yaptık Filtreleme
# print(q7.head())
# q7=f[f["Genre"]=="Misc"].sort_values("Year") ### yıllara göre filtreleme yapıyor
# print(q7.head())


# # VERİ TEMİZLEME / DÜZENLEME

# z=f[["Other_Sales","NA_Sales","EU_Sales"]].isnull().head() ### Other_sales,NA_Sales,EU_Sales ilk 5 veride 0 dan farklı ise False ,0 ise True Döndürecektir .
# print(z)
# f["JP_Sales"]=f["JP_Sales"].interpolate().astype(float) # eksik yerleri interpolate ile  kendisi veri setine göre  değerler atadı  daha uygun temizleme işlemi yaptı float olarak atadı.
# print(f)


# print(f.isnull().sum().sort_values(ascending=False)) ### eksik yerleri temizlemiş olduk 2.yol
# print(f)


# print(f["Other_Sales"].sum) ### sum fonksiyonu toplamlarını alır
# print(f)
# print(f.at[3,"EU_Sales"]) ##at hazır fonksiyonu 3.satır EU_Sales indeksine ait değerleri verir



# # GLOBAL(toptan) satışın  Yıla göre  ilişkisini görselleştirerek inceleyelim

# f.plot(x="Year",y='Global_Sales',kind='hist',bins=20)  ##2 veriyi bir grafikte birleştiricez veriler x-y olarak atanacak kind yani grafik çeşitliliği histogram grafiği olucak
# plt.xlabel('Year')
# plt.xlim(0,50) ##x ekseni sınırlandırma 0 -50
# plt.ylabel('Global_Sales')
# plt.ylim(0,10) ## y ekseni sınırlandırma 0,10
# plt.title("Toptan satışın Yıllara göre ilişkisi") ##başlık
# plt.show()
# print(f)
# ##toptan satış ve diğer satış arasındaki ilişkiyi görselleştirerek incelemiş olduk
# ## SONUÇ: 0-15,25-28 ve 43  yıllar arası toptan satışın olmadığını grafik sayesinde öğrenmiş olduk

# # Diğer Satışların (Other_Sales) Year yıllara göre satış analizini inceleyelim

# f.plot(x="Year",y='Other_Sales',kind="hist",bins=40)
# plt.xlabel('Year')
# plt.xlim(0,10)
# plt.ylabel('Other_Sales')
# plt.ylim(0,10)
# plt.title("Diğer  satışın Yıllara göre ilişkisi")
# plt.show()
# print(f)
# ## 4.5- 8  arası yıllar diğer satışlar olmadığı görülmüştür


# f['Year'] = f['Other_Sales']/f['Global_Sales']
# grafik=f['Year'].sort_values(ascending=False).head(20)
# grafik.plot(x='Other_Sales',y='Year',kind='bar')
# plt.show()


# ## SATIŞ VERİLERİN YILA GÖRE İNCELENMESİ

# x=f["Other_Sales"].sort_values() ###verileri sort_values ile sıralama yapıyoruz düzgün görsel elde etmek için
# print(x)
# y=f["Global_Sales"].sort_values()
# z=f["Year"].sort_values()
# x1=f["JP_Sales"].sort_values()
# y1=f["EU_Sales"].sort_values()

# plt.title("YILLARA GÖRE SATIŞ")
# plt.subplot(2,2,2) ###2*2 4 lük alan açar  sondaki grafik yeridir.
# plt.plot(z,x,color = "green",marker ="o",markersize=2,markerfacecolor ="red",markeredgecolor = "yellow",markeredgewidth = 4)
# plt.title("Diğer satışlar ın yıllara göre değişimi")
# plt.xlabel("yıllar")
# plt.ylabel("Diğer Satışlar")
# plt.tight_layout()

# plt.subplot(2,2,1)
# plt.title("Toptan satışın yıllara göre incelenmesi")
# plt.plot(z,y,color = "red",marker ="o",markersize=2,markerfacecolor ="blue",markeredgecolor = "green",markeredgewidth = 2)
# plt.xlabel("yıllar")
# plt.ylabel("Toptan Satış")
# plt.tight_layout()

# plt.subplot(2,2,3)
# plt.title("JP satışın yıllara göre incelenmesi")
# plt.plot(z,x1,color = "yellow",marker ="o",markersize=1,markerfacecolor ="silver",markeredgecolor = "orange",markeredgewidth = 5)
# plt.xlabel("yıllar")
# plt.ylabel(" JP satış")
# plt.tight_layout()

# plt.subplot(2,2,4)
# plt.title("Avrupa satışın yıllara göre incelenmesi")
# plt.plot(z,y1,color = "purple",marker ="o",markersize=2,markerfacecolor ="blue",markeredgecolor = "red",markeredgewidth = 3)
# plt.xlabel("yıllar")
# plt.ylabel(" EU(avrupa) satış")
# plt.tight_layout() ###Grafikler arası boşluk açar 
# plt.show()
# ###Genel olarak JP_Sales satışın Yıllara göre Başarılı bir satış yapıldığı görülüyor 






# # ## verilerin ORTALAMA/MOD/MEDYAN DEĞERLERİ BULALIM

# K=f.mean(axis=1,skipna=None).sort_values() ###ortalama axis=1 columns içindir skipna atlama numarasıdır
# O=f.median(axis=1,skipna=None).sort_values() ##medyan yani orta değer
# U=f.std(axis=1,skipna=None).sort_values() ## standart sapma
# M=f.sum(axis=1,skipna=None).sort_values() ## verilerin toplamları
# print("ortalama\n {} medyan \n {}  standartsapma \n{} toplam  verileri {} ".format(K,O,U,M))

# fig,axes = plt.subplots(nrows = 2, ncols = 1) ## 2satır 1 kolon luk alan oluşturur grafik
# axes[0].plot(M,O,color = "blue",marker ="o",markersize=3,markerfacecolor ="black",markeredgecolor = "black",markeredgewidth = 5)
# axes[0].set_title("toplam veri-Medyan Grafiği")
# axes[0].set_xlabel("Toplam")
# axes[0].set_ylabel("Medyan ")
# plt.tight_layout()
# ## Toplam değer le Medyanın yaklaşık doğru orantılı olarak arttığı görülüyor


# axes[1].plot(U,K,color = "red",marker ="o",markersize=3,markerfacecolor ="yellow",markeredgecolor = "black",markeredgewidth = 3)
# axes[1].set_title("Ortalama -standart Sapma")
# axes[1].set_xlabel("Ortalama")
# axes[1].set_ylabel("Standart sapma")
# plt.tight_layout()
# plt.show()
# ##  her bir verinin aritmetik ortalamadan çıkarılıp karesini alırnır toplanıp veri sayısının 1 eksiğine bölünür ve karekökü alınır
# ## dolayısıyla artitmetik ortalamanın artmması ile d.o olarak standart sapma artar


# ## Nitel veriler sayısal analiz lerde sıkıntı çıkarabilir ozaman Dummies Değişkenleri ile sayısal verilere çeviririz

# f_dummy=pd.get_dummies(f["Publisher"]) ##publisher verileri dummies değişkenlerine çevirdik
# f=pd.concat([f,f_dummy],axis=1) ##pd.concat ile dummy değişkenlerini verimize aktarıp güncelledik
# print(f)
# a=f["Microsoft Game Studios"].sort_values() ##değerlere göre sıralama yaptık
# b=f["Take-Two Interactive"].sort_values()
# c=f["Sony Computer Entertainment"].sort_values()
# d=f["Activision"].sort_values()

# print(f_dummy)
# print(f)


# ## GÖRSELLEŞTİRME

# plt.title("YAYIMCI ŞİRKETLER")
# plt.subplot(2,2,1) 
# plt.plot(a,b)
# plt.xlabel("Microsoft Game Studios Yayımcısı")
# plt.ylabel("Take-Two Interactive yayımcı")
# plt.tight_layout()

# plt.subplot(2,2,2) 
# plt.plot(c,d)
# plt.xlabel(" Sony Computer Entertainment Yayımcısı")
# plt.ylabel("ACTİVİSİON yayımcı")
# plt.tight_layout()
# plt.show()

# print(f_dummy)
# print(f)


