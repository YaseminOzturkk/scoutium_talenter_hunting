########################################################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma#
########################################################

# İş Problemi

"""Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
(average, highlighted) oyuncu olduğunu tahminleme"""

# Veri Seti Hikayesi
"""Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır."""


# scoutium_attributes.csv Variables
#task_response_id Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id İlgili maçın id'si
# evaluator_id Değerlendiricinin(scout'un) id'si
# player_id İlgili oyuncunun id'si
# position_id İlgili oyuncunun o maçta oynadığı pozisyonun id’si
    #1: Kaleci
    #2: Stoper
    #3: Sağ bek
    #4: Sol bek
    #5: Defansif orta saha
    #6: Merkez orta saha
    #7: Sağ kanat
    #8: Sol kanat
    #9: Ofansif orta saha
    #10: Forvet
#analysis_id Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
#attribute_id Oyuncuların değerlendirildiği her bir özelliğin id'si
#attribute_value Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv Variables
#task_response_id Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id İlgili maçın id'si
#evaluator_id Değerlendiricinin(scout'un) id'si
#player_id İlgili oyuncunun id'si
#potential_label Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

# Gereklilikler

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

scoutium_attributes = pd.read_csv("C:/Users/yasmi/Desktop/machine_learning/Scoutium/scoutium_attributes.csv", sep=';')
scoutium_potential_labels = pd.read_csv("C:/Users/yasmi/Desktop/machine_learning/Scoutium/scoutium_potential_labels.csv", sep=';')

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)


df = pd.merge(scoutium_attributes, scoutium_potential_labels, on=['task_response_id','match_id', 'evaluator_id', 'player_id'],
how='right')


#Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df["position_id"].value_counts()
df = df[df["position_id"] != 1]

#Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df["potential_label"].value_counts()
df = df[df["potential_label"] != 'below_average']

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.
    # Adım 5.1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
    # “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
    #Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.


final_df = df.pivot_table(values="attribute_value", index=["player_id", "position_id", "potential_label"], columns="attribute_id")
final_df = final_df.reset_index()  # inplace parametresi kullanılmadı, reset_index'in döndürdüğü değeri final_df değişkenine atadık
final_df.columns = final_df.columns.astype(str)


################################################
# 1. Exploratory Data Analysis
################################################

# Genel resmi inceleyelim.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)
check_df(final_df)

# Kategorik, numerik, numerik görünümlü kategorik ve kardinal değişkenleri yakalayalım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(final_df, cat_th=10, car_th=20)

cat_but_car = "player_id"
num_cols = [col for col in num_cols if col not in "player_id"]
cat_cols = [col for col in cat_cols if col not in ['4324', '4328', '4352', '4357', '4423']]
num_cols = num_cols + ['4324', '4328', '4352', '4357', '4423']


# Kategorik değişken analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
# Numerik değişken analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# Hedef değişkeni numerik değişkenler ile analiz
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# Hedef değişkeni kategorik değişkenler ile analiz
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# Korelasyon analizi
def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

# Fonksiyonu doğru değişkenle çağırın
correlation_matrix(final_df, num_cols)


# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(final_df, col, plot=True)

# Sayısal değişkenlerin incelenmesi
final_df[num_cols].describe().T

for col in num_cols:
    num_summary(final_df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(final_df, num_cols)

# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(final_df, "potential_label", col)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Alt ve üst limit belirleme
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değerleri alt ve üst limit değerleriyle değiştirme
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı değer olup olmadığına bakma
def check_outlier(dataframe, col_name, q1=0.1, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(final_df, col, q1=0.1, q3=0.99))



# Kategorik değişkenlere encoding işlemi uygulama
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(final_df, col)

# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
num_cols = num_cols + ['4324', '4328', '4352', '4357', '4423']

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız
scaler = StandardScaler()
final_df[num_cols] = scaler.fit_transform(final_df[num_cols])

# Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

# Bağımlı ve bağımsız değişkenleri seçelim
y = final_df["potential_label"]
X = final_df.drop(["potential_label", "player_id"], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

"""def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modelini oluşturma
model = LGBMClassifier()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Olasılıkları al

# Performans metriklerini yazdırma
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"Precision Score: {precision_score(y_test, y_pred)}")
print(f"Recall Score: {recall_score(y_test, y_pred)}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")


"""
ROC-AUC Score: 0.8636363636363636
F1 Score: 0.6666666666666665
Precision Score: 0.8571428571428571
Recall Score: 0.5454545454545454
Accuracy Score: 0.8909090909090909
"""


def plot_importance(model, features, num=10, save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")

# LightGBM modeli üzerinde feature importance çizdirin
model = LGBMClassifier()
model.fit(X, y)
plot_importance(model, X)
