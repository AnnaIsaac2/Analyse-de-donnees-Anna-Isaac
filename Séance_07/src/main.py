#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
data = pd.DataFrame(ouvrirUnFichier("./data/resultats-elections-presidentielles-2022-1er-tour.csv"))
# print(data)

# "Code_du_département"
# "Libellé_du_département"
# "Inscrits"
# "Abstentions"
# "Votants"
# "Blancs"
# "Nuls"
# "Exprimés"
# "Nathalie_ARTHAUD"
# "Fabien_ROUSSEL"
# "Emmanuel_MACRON"
# "Jean_LASSALLE"
# "Marine_LE_PEN"
# "Éric_ZEMMOUR"
# "Jean_Luc_MÉLENCHON"
# "Anne_HIDALGO"
# "Yannick_JADOT"
# "Valérie_PÉCRESSE"
# "Philippe_POUTOU"
# "Nicolas_DUPONT_AIGNAN"

# Création du tableau de contingence
# Contrairement à l'usage, vous ne devez pas créer de tableau croisé dynamique, puisque le fichier est déjà un tableau de contingence
nomdescolonnes = ["Libellé_du_département", "Nathalie_ARTHAUD", "Fabien_ROUSSEL", "Emmanuel_MACRON", "Jean_LASSALLE", "Marine_LE_PEN", "Éric_ZEMMOUR", "Jean_Luc_MÉLENCHON", "Anne_HIDALGO", "Yannick_JADOT", "Valérie_PÉCRESSE", "Philippe_POUTOU", "Nicolas_DUPONT_AIGNAN"]
# print(nomdescolonnes)
dictionnaire = {}
for element in range(1, len(nomdescolonnes)):
    dictionnaire[nomdescolonnes[element]] = data[nomdescolonnes[element]]
tableauDeContingence = tableauDeContingence(data[nomdescolonnes[0]], dictionnaire)
# print(tableauDeContingence)

# Question 1
print("Question 1")
margesColonnes = sommeDesColonnes(tableauDeContingence)
margesLignes = sommeDesLignes(tableauDeContingence)

print("Marges des colonnes :", margesColonnes)
print("Marges des lignes :", margesLignes)

totalColonnes = np.sum(margesColonnes)
totalLignes = np.sum(margesLignes)

print("Total des colonnes :", totalColonnes)
print("Total des lignes :", totalLignes)

# Question 2
print("Question 2")
if totalLignes == totalColonnes:
    print("Les deux totaux sont identiques.")
else:
    print("Les deux totaux sont différents.")

# Question 3
print("Question 3")
resultatChi2 = scipy.stats.chi2_contingency(
    tableauDeContingence
)

chi2, p_value, degresLiberte, frequencesAttendues = resultatChi2

print("Statistique du chi2 :", chi2)
print("p-value :", p_value)
print("Degrés de liberté :", degresLiberte)

if p_value < 0.05:
    print("Il existe une liaison statistiquement significative.")
else:
    print("Aucune liaison statistiquement significative n'est mise en évidence.")

# Question 4
print("Question 4")
phi2 = chi2 / totalLignes

print("Intensité de liaison phi2 de Pearson :", phi2)

# Question bonus
print("Question bonus")
echantillons = ouvrirUnFichier(
    "./data/Echantillonnage-100-Echantillons.csv"
)

resultatANOVA = scipy.stats.f_oneway(
    echantillons["Pour"],
    echantillons["Contre"],
    echantillons["Sans opinion"]
)

print("Statistique F :", resultatANOVA[0])
print("p-value :", resultatANOVA[1])
print("Bonus - A.F.C.")

tableau = tableauDeContingence.to_numpy(dtype=float)

total = tableau.sum()
frequences = tableau / total

margesLignesAFC = frequences.sum(axis=1)
margesColonnesAFC = frequences.sum(axis=0)

attendues = np.outer(
    margesLignesAFC,
    margesColonnesAFC
)

residus = (
    frequences - attendues
) / np.sqrt(attendues)

U, valeursSingulieres, Vt = np.linalg.svd(
    residus,
    full_matrices=False
)

valeursPropres = valeursSingulieres ** 2
inertieTotale = valeursPropres.sum()

print("Inertie totale :", inertieTotale)
print("Valeur propre axe 1 :", valeursPropres[0])
print("Valeur propre axe 2 :", valeursPropres[1])

print(
    "Inertie expliquée par l'axe 1 (%) :",
    valeursPropres[0] / inertieTotale * 100
)

print(
    "Inertie expliquée par l'axe 2 (%) :",
    valeursPropres[1] / inertieTotale * 100
)

# Calcul des coordonnées factorielles

coordDepartements = (
    U[:, :2] * valeursSingulieres[:2]
) / np.sqrt(margesLignesAFC[:, None])

coordCandidats = (
    Vt[:2, :].T * valeursSingulieres[:2]
) / np.sqrt(margesColonnesAFC[:, None])

# Création du graphique
plt.figure(figsize=(14, 10))

plt.scatter(
    coordDepartements[:, 0],
    coordDepartements[:, 1],
    label="Départements",
    alpha=0.6
)

plt.scatter(
    coordCandidats[:, 0],
    coordCandidats[:, 1],
    label="Candidats",
    marker="x",
    s=100
)

for i, nom in enumerate(tableauDeContingence.index):
    plt.annotate(
        nom,
        (coordDepartements[i, 0], coordDepartements[i, 1]),
        fontsize=6
    )

for j, nom in enumerate(tableauDeContingence.columns):
    plt.annotate(
        nom,
        (coordCandidats[j, 0], coordCandidats[j, 1]),
        fontsize=8
    )

plt.axhline(0)
plt.axvline(0)

plt.xlabel("Axe 1 (47,46 %)")
plt.ylabel("Axe 2 (22,27 %)")
plt.title("A.F.C. des résultats du premier tour de l'élection présidentielle de 2022")
plt.legend()
plt.grid(True)

plt.savefig("afc_elections_2022.png", dpi=300, bbox_inches="tight")
plt.show()