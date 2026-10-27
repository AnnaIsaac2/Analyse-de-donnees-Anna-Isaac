#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
donnees = ouvrirUnFichier("data/pib-vs-energie.csv")
donneesOriginales = donnees.copy()
print(donnees[["PIB_2022", "Utilisation_d_energie_2022"]])

# Question 2
print("Question 2")
donneesCompletes = []

for i in range(len(donnees)):
    pib = donnees["PIB_2022"][i]
    energie = donnees["Utilisation_d_energie_2022"][i]

    if not np.isnan(pib) and not np.isnan(energie):
        donneesCompletes.append([pib, energie])

donnees = pd.DataFrame(
    donneesCompletes,
    columns=["PIB_2022", "Utilisation_d_energie_2022"]
)

print(donnees)
print("Nombre de couples complets :", len(donnees))

# Question 3
print("Question 3")
x = donnees["Utilisation_d_energie_2022"]
y = donnees["PIB_2022"]

regression = scipy.stats.linregress(x, y)

print("Pente :", regression.slope)
print("Ordonnée à l'origine :", regression.intercept)
print("Coefficient de corrélation :", regression.rvalue)
print("Coefficient de détermination :", regression.rvalue ** 2)

# Question 4
print("Question 4")
correlation = donnees["PIB_2022"].corr(
    donnees["Utilisation_d_energie_2022"]
)

print("Corrélation de Pearson :", correlation)

# Question 5
print("Question 5")
plt.scatter(x, y, label="Territoires")

xDroite = np.linspace(x.min(), x.max(), 100)

plt.plot(
    xDroite,
    regression.slope * xDroite + regression.intercept,
    color="red",
    label="Droite de régression"
)

plt.xlabel("Consommation énergétique (kg équivalent pétrole)")
plt.ylabel("PIB (dollars courants)")
plt.title("PIB et consommation énergétique en 2022")
plt.legend()
plt.savefig("regression-2022.png")
plt.show()

# Question 6
print("Question 6")

# Bonus : calculer la régression pour chaque année de 1962 à 2022
print("Bonus")

resultats = []

for annee in range(1962, 2023):
    colonnePIB = "PIB_" + str(annee)
    colonneEnergie = "Utilisation_d_energie_" + str(annee)

    couplesComplets = donneesOriginales[
        [colonnePIB, colonneEnergie]
    ].copy()

    couplesComplets[colonnePIB] = pd.to_numeric(
        couplesComplets[colonnePIB], errors="coerce"
    )
    couplesComplets[colonneEnergie] = pd.to_numeric(
        couplesComplets[colonneEnergie], errors="coerce"
    )

    couplesComplets = couplesComplets.dropna()

    if len(couplesComplets) >= 2:
        x = couplesComplets[colonneEnergie]
        y = couplesComplets[colonnePIB]

        regression = scipy.stats.linregress(x, y)

        resultats.append({
            "Annee": annee,
            "Nombre_de_couples": len(couplesComplets),
            "Pente": regression.slope,
            "Ordonnee_a_l_origine": regression.intercept,
            "Correlation_de_Pearson": regression.rvalue,
            "Coefficient_de_determination": regression.rvalue ** 2
        })

        print("Année :", annee, "- régression calculée")

# Enregistrer tous les résultats dans un fichier CSV
tableauResultats = pd.DataFrame(resultats)
tableauResultats.to_csv(
    "resultats-regressions-1962-2022.csv",
    index=False,
    encoding="utf-8"
)

print("Résultats enregistrés dans resultats-regressions-1962-2022.csv")
