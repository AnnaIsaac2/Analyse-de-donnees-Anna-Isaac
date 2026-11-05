#coding:utf8
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
import scipy.stats
import prince
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

os.makedirs("img", exist_ok=True)

# Question 1
print("Question 1 : Analyse en composantes principales")
df = ouvrirUnFichier("data/SIPRI-1949-2025.csv")
print(df.shape)
# Question 1b
print("Question 1b")
individus = df["Code ISO"]
num = df.drop(columns=["Code ISO"])
num = num.drop(columns=["Country"])
num = num.drop(columns=["État"])
num = num.drop(columns=["Continent"])
print(num.shape)

# "Code ISO" : code ISO du territoire
# "Country" : territory name
# "État" : nom du territoire
# "Continent" : Continent
# "1949","1950","1951","1952","1953","1954","1955","1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025" : 77 colonnes des différentes années

# Centrer-réduire
# Question 1c
print("Question 1c")
num = num.fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(num)
print(X[:2, :5])
# ACP

# Question 1d
print("Question 1d")
p = 77
pca = PCA(n_components=p)
pca.fit(X)

noms_facteurs = []
for i in range(p):
    noms_facteurs.append("F" + str(i + 1))

# Affichage de l'A.C.P.
print(pca)

# Question 1e
print("Question 1e")
print(pca.explained_variance_)
# Variance expliquée en pourcentage
print(pca.explained_variance_ratio_ * 100)
# Calcul des valeurs propres
valeurs_propres = pca.explained_variance_
print(valeurs_propres)

# Question 1f
print("Question 1f")
resultat = pd.DataFrame(pca.components_, index=noms_facteurs, columns=num.columns)
print(resultat)
# Coordonnées des individus
coord_ind = pca.transform(X)
df_coord = pd.DataFrame(coord_ind, index=individus, columns=noms_facteurs)
print(df_coord)

# Question 1g
print("Question 1g")
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(df_coord["F1"], df_coord["F2"])
for code, x, y in zip(df_coord.index, df_coord["F1"], df_coord["F2"]):
    ax.annotate(code, (x, y), fontsize=7)
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("F1 (" + str(round(pca.explained_variance_ratio_[0] * 100, 1)) + " %)")
ax.set_ylabel("F2 (" + str(round(pca.explained_variance_ratio_[1] * 100, 1)) + " %)")
ax.set_title("Mapping des individus (F1, F2)")
plt.savefig("img/acp_individus.png", dpi=150)
plt.close()

# Visualisation des individus

# Contribution des individus
# Question 1h
print("Question 1h")

n = X.shape[0]
print(n)

contrib = coord_ind ** 2 / ((n - 1) * valeurs_propres) * 100

noms_contrib = []
for i in range(p):
    noms_contrib.append("CTR_F" + str(i + 1))

df_contrib = pd.DataFrame(contrib, index=individus, columns=noms_contrib)
print(df_contrib)

cos2 = coord_ind ** 2 / (coord_ind ** 2).sum(axis=1, keepdims=True)

noms_cos2 = []
for i in range(p):
    noms_cos2.append("COS2_F" + str(i + 1))

df_cos2 = pd.DataFrame(cos2, index=individus, columns=noms_cos2)
print(df_cos2)

print(df_contrib["CTR_F1"].sort_values(ascending=False).head(5))
# Les 5 individus qui contribuent le plus au facteur 2
print("Les 5 plus fortes contributions sur F2")
print(df_contrib["CTR_F2"].sort_values(ascending=False).head(5))

# Les 5 individus les mieux représentés sur F1
print("Les 5 meilleures qualités de projection sur F1")
print(df_cos2["COS2_F1"].sort_values(ascending=False).head(5))

# Les 5 individus les mieux représentés sur F2
print("Les 5 meilleures qualités de projection sur F2")
print(df_cos2["COS2_F2"].sort_values(ascending=False).head(5))

# Question 1i
print("Question 1i")
# Racine carrée des valeurs propres
sqrt_vp = np.sqrt(valeurs_propres)
# Matrice vide pour avoir les coordonnées
coord_var = np.zeros((p, p))
for k in range(p):
    coord_var[:, k] = pca.components_[k, :] * sqrt_vp[k]
# Création d'un DataFrame
df_var = pd.DataFrame(coord_var, index=num.columns, columns=noms_facteurs)
print(df_var)
# Visualisation du cercle de corrélation
fig, ax = plt.subplots(figsize=(9, 9))
ax.add_patch(plt.Circle((0, 0), 1, color="grey", fill=False))
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
for annee, x, y in zip(df_var.index, df_var["F1"], df_var["F2"]):
    ax.arrow(0, 0, x, y, head_width=0.015, color="tab:blue", alpha=0.6)
    ax.annotate(annee, (x, y), fontsize=7)
ax.set_xlabel("F1 (" + str(round(pca.explained_variance_ratio_[0] * 100, 1)) + " %)")
ax.set_ylabel("F2 (" + str(round(pca.explained_variance_ratio_[1] * 100, 1)) + " %)")
ax.set_title("Cercle de corrélation")
plt.savefig("img/acp_cercle_correlation.png", dpi=150)
plt.close()

# Question 2
print("Question 2 : Analyse factorielle de correspondances multiples")
# Analyse factorielle de correspondances multiples
# Question 2a
chiens = ouvrirUnFichier("data/chiens.csv")
print(chiens.shape)

# Question 2b
races = chiens["Race"]
variables = chiens[["Taille", "Poids", "Vitesse", "Intelligence", "Affection", "Agressivité", "Fonction", "Origine"]]
print(variables.head())

# Question 2c
print("Question 2c")
# Calcul du tableau disjonctif complet
tdc = pd.get_dummies(variables)
tdc.index = races
print(tdc.shape)
print(tdc.head())

# Question 2d
print("Question 2d")
print("version de prince :", prince.__version__)
mca = prince.MCA(n_components=8, random_state=42)
mca = mca.fit(tdc)
print(mca)

# Question 2e
print("Question 2e")
# Calcul et affichage des valeurs propres
valeurs_propres_mca = np.array(mca.eigenvalues_)
print(valeurs_propres_mca)

# Variance expliquée en pourcentage
print(mca.percentage_of_variance_)

# Question 2f
print("Question 2f")
# Coordonnées des lignes et des colonnes
coord_lignes = mca.row_coordinates(tdc)
print(coord_lignes)
coord_colonnes = mca.column_coordinates(tdc)
print(coord_colonnes)

# Question 2g
print("Question 2g")
# Visualiser les deux premiers facteurs
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(coord_lignes.iloc[:, 0], coord_lignes.iloc[:, 1], color="tab:blue", label="Races")
for nom, x, y in zip(coord_lignes.index, coord_lignes.iloc[:, 0], coord_lignes.iloc[:, 1]):
    ax.annotate(nom, (x, y), fontsize=8, color="tab:blue")
ax.scatter(coord_colonnes.iloc[:, 0], coord_colonnes.iloc[:, 1], color="tab:red", marker="^", label="Modalités")
for nom, x, y in zip(coord_colonnes.index, coord_colonnes.iloc[:, 0], coord_colonnes.iloc[:, 1]):
    ax.annotate(nom, (x, y), fontsize=8, color="tab:red")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Facteur 1")
ax.set_ylabel("Facteur 2")
ax.legend()
ax.set_title("A.C.M. : mapping des deux premiers facteurs")
plt.savefig("img/acm_mapping.png", dpi=150)
plt.close()

# Question 2h
print("Question 2h")
Z = tdc.values.astype(float)
masse_lignes = Z.sum(axis=1) / Z.sum()
masse_colonnes = Z.sum(axis=0) / Z.sum()
vp = np.array(mca.eigenvalues_)
# Calculer la contribution de l'A.C.M. avec `Prince` des lignes
contrib_lignes = (coord_lignes.values ** 2) * masse_lignes[:, None] / vp
noms_f = []
for i in range(8):
    noms_f.append("CTR_F" + str(i + 1))
df_contrib_lignes = pd.DataFrame(contrib_lignes * 100, index=coord_lignes.index, columns=noms_f)
# Calculer la contribution de l'A.C.M. avec `Prince` des colonnes
contrib_colonnes = (coord_colonnes.values ** 2) * masse_colonnes[:, None] / vp
df_contrib_colonnes = pd.DataFrame(contrib_colonnes * 100, index=coord_colonnes.index, columns=noms_f)
print(df_contrib_colonnes)
d2_lignes = (((Z / 8 - masse_colonnes) ** 2) / masse_colonnes).sum(axis=1)
cos2_lignes = (coord_lignes.values ** 2) / d2_lignes[:, None]
noms_c = []
for i in range(8):
    noms_c.append("COS2_F" + str(i + 1))
df_cos2_lignes = pd.DataFrame(cos2_lignes, index=coord_lignes.index, columns=noms_c)
print(df_cos2_lignes)
profils = Z / Z.sum(axis=0)
d2_colonnes = (((profils - masse_lignes[:, None]) ** 2) / masse_lignes[:, None]).sum(axis=0)
cos2_colonnes = (coord_colonnes.values ** 2) / d2_colonnes[:, None]
df_cos2_colonnes = pd.DataFrame(cos2_colonnes, index=coord_colonnes.index, columns=noms_c)
print(df_cos2_colonnes)
# Affichage des contributions des lignes
print("Contributions des lignes")
print(df_contrib_lignes)

# Affichage des contributions des colonnes
print("Contributions des colonnes")
print(df_contrib_colonnes)

# Question bonus
print("Question bonus")
k_axes = 5
coord_cah = coord_ind[:, :k_axes]
Z_cah = linkage(coord_cah, method="ward")
plt.figure(figsize=(16, 7))
dendrogram(Z_cah, labels=individus.values, leaf_rotation=90, leaf_font_size=6)
plt.title("C.A.H. (Ward) sur les coordonnées de l'A.C.P.")
plt.tight_layout()
plt.savefig("img/cah_dendrogramme.png", dpi=150)
plt.close()
groupes = fcluster(Z_cah, t=4, criterion="maxclust")
resultat_cah = pd.Series(groupes, index=individus).sort_values()
print(resultat_cah)
print(resultat_cah.value_counts())
plt.figure(figsize=(12, 10))
plt.scatter(coord_ind[:, 0], coord_ind[:, 1], c=groupes)
for code, x, y in zip(individus, coord_ind[:, 0], coord_ind[:, 1]):
    plt.annotate(code, (x, y), fontsize=7)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Individus colorés par classe de la C.A.H.")
plt.savefig("img/cah_classes.png", dpi=150)
plt.close()
# Effectifs de chaque classe
print("Nombre d'individus par classe")
print(resultat_cah.value_counts().sort_index())

# Moyennes des coordonnées des individus par classe
df_classes = pd.DataFrame(coord_cah, index=individus)
df_classes["Classe"] = groupes

print("Caractéristiques des classes")
print(df_classes.groupby("Classe").mean())