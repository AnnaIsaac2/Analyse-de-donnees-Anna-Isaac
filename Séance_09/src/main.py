#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import scipy
import scipy.stats
import math
import os

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
temperature = ouvrirUnFichier("data/temperature.csv")

# Question 1 - Premier tiret
print("Question 1 - Premier tiret")
print(temperature.corr())

# Question 1 - Deuxième tiret
print("Question 1 - Deuxième tiret")
print(temperature.describe())

# Question 1 - Troisième tiret
print("Question 1 - Troisième tiret")
villes = temperature["Ville"]
y = temperature["Température_en_janvier"]
X = temperature[["Latitude", "Longitude", "Altitude"]]
print(X.head())

# Question 1 - Quatrième tiret
print("Question 1 - Quatrième tiret")
X_const = sm.add_constant(X)
modele = OLS(y, X_const).fit()

# Question 1 - Cinquième tiret
print("Question 1 - Cinquième tiret")
print(modele.params)
print(modele.rsquared)
print(modele.pvalues)
print(modele.summary())

# Question 1 - Sixième tiret
print("Question 1 - Sixième tiret")
reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.intercept_)

# Question 1 - Septième tiret
print("Question 1 - Septième tiret")
resultats = pd.DataFrame({"Ville": villes, "observé": y,
                          "prédit": modele.fittedvalues,
                          "résidu": modele.resid})
print(resultats.sort_values("résidu"))

# Question 2
print("Question 2")
geomarketing = ouvrirUnFichier("data/geomarketing.csv")
ca = geomarketing["ca"]

X = geomarketing[["surface_totale", "potentiel_Z20",
                 "nb_primaire_Z10", "nb_primaire_Z20",
                 "nb_gsa_Z10", "nb_pharmacie_Z5",
                 "nb_conc2_Z10", "nb_conc2_Z20",
                 "P10_POP_Z15", "P10_MEN_Z10"]]

# Question 2 - Premier tiret
print("Question 2 - Premier tiret")
geomarketing = ouvrirUnFichier("data/geomarketing.csv")
print(geomarketing.head())

# Question 2 - Deuxième tiret
print("Question 2 - Deuxième tiret")
ca = geomarketing["ca"]

X = geomarketing[["surface_totale", "potentiel_Z20",
                 "nb_primaire_Z10", "nb_primaire_Z20",
                 "nb_gsa_Z10", "nb_pharmacie_Z5",
                 "nb_conc2_Z10", "nb_conc2_Z20",
                 "P10_POP_Z15", "P10_MEN_Z10"]]

print(X.head())

# Question 2 - Troisième tiret
print("Question 2 - Troisième tiret")
X_const = sm.add_constant(X)
modele = OLS(ca, X_const).fit()

print(modele.params)
print(modele.rsquared)
print(modele.pvalues)
print(modele.summary())

# Question 2 - Quatrième tiret
print("Question 2 - Quatrième tiret")
print("Le R2 du modèle est de :", modele.rsquared)
print("Le R2 ajusté du modèle est de :", modele.rsquared_adj)
print("La probabilité du test de Fisher est de :", modele.f_pvalue)

# Bonus
print("Bonus")

# Bonus - Premier tiret
print("Bonus - Premier tiret")

X_bonus = geomarketing.drop(columns=["enseigne", "ca"])
X_bonus = X_bonus.select_dtypes(include=[np.number])

print(X_bonus.head())
print(X_bonus.shape)

# Bonus - Deuxième tiret
print("Bonus - Deuxième tiret")

X_bonus_const = sm.add_constant(X_bonus)
modele_bonus = OLS(ca, X_bonus_const).fit()

print("R2 :", modele_bonus.rsquared)
print("R2 ajusté :", modele_bonus.rsquared_adj)
print("Probabilité de Fisher :", modele_bonus.f_pvalue)
print("P-values :")
print(modele_bonus.pvalues)