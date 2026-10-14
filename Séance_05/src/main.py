#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding='utf-8') as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False and pop[element] != 0.0:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

def classementPays(ordre1, ordre2):
    rang2 = {}
    for rang, etat in ordre2:
        rang2[etat] = rang
    classement = []
    for rang, etat in ordre1:
        if etat in rang2:
            classement.append([rang, rang2[etat], etat])
    classement.sort()
    return classement

def isolerColonnes(classement):
    rang1 = []
    rang2 = []
    for element in classement:
        rang1.append(element[0])
        rang2.append(element[1])
    return rang1, rang2

def analyseRangs(ordre1, ordre2):
    rang1, rang2 = isolerColonnes(classementPays(ordre1, ordre2))
    if len(rang1) < 3:
        return np.nan, np.nan, len(rang1)
    spearman = scipy.stats.spearmanr(rang1, rang2)[0]
    kendall = scipy.stats.kendalltau(rang1, rang2)[0]
    return spearman, kendall, len(rang1)

def comparerToutesLesAnnees(donnees, annees, prefixe, colEtat):
    etats = list(donnees[colEtat])
    classements = []
    for annee in annees:
        valeurs = list(pd.to_numeric(donnees[prefixe + annee], errors="coerce"))
        classements.append(ordrePopulation(valeurs, etats))
    resultats = []
    for i in range(len(annees)):
        for j in range(i + 1, len(annees)):
            s, k, n = analyseRangs(classements[i], classements[j])
            resultats.append([annees[i], annees[j], s, k, n])
    return pd.DataFrame(resultats, columns=["Année 1", "Année 2", "Spearman", "Kendall", "N États"])

# Question 2
print("Question 2")
depenses = ouvrirUnFichier("data/SIPRI-1949-2025.csv")

# Question 3
print("Question 3")

depenses2025 = list(depenses["2025"])

depenses2025SansNaN = []

for element in depenses2025:
    if np.isnan(element) == False and element != 0.0:
        depenses2025SansNaN.append(element)

# "Code ISO" : code ISO du territoire
# "Country" : territory name
# "État" : nom du territoire
# "Continent" : Continent
# "1949","1950","1951","1952","1953","1954","1955","1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025" : 77 colonnes des différentes années

# Question 4
print("Question 4")

depenses2025SansNaN = ordreDecroissant(depenses2025SansNaN)

# Question 5
print("Question 5")

plt.plot(range(1, len(depenses2025SansNaN) + 1), depenses2025SansNaN)
plt.xlabel("Rang")
plt.ylabel("Dépenses militaires (milliards USD courants)")
plt.title("Classement des pays par dépenses militaires en 2025")
plt.savefig("depenses2025.png")
plt.close()

# Question 6
print("Question 6")


logDepenses2025 = conversionLog(depenses2025SansNaN)
logRang = conversionLog(list(range(1, len(depenses2025SansNaN) + 1)))

plt.plot(logRang, logDepenses2025)
plt.xlabel("log(Rang)")
plt.ylabel("log(Dépenses militaires)")
plt.savefig("depenses2025_log.png")
plt.close()

# Question 7
print("Question 7")

# Non, pas directement sur une seule série. Le rang est calculé à partir des
# dépenses elles-mêmes : corréler le rang avec les dépenses donnerait une
# relation parfaite (-1) sans intérêt statistique. Un test sur les rangs
# (Spearman, Kendall) n'est possible que pour comparer deux classements
# distincts, par exemple celui de deux années. On peut en revanche examiner
# le graphique log-log, où une droite suggère une loi rang-taille.

# Question 9
print("Question 9")

etats = ouvrirUnFichier("data/Le-Monde-HS-Etats-du-monde-2007-2026.csv")

# Question 10
print("Question 10")

etat = list(etats["État"])

pop2007 = list(pd.to_numeric(etats["Pop 2007"], errors="coerce"))
pop2026 = list(pd.to_numeric(etats["Pop 2026"], errors="coerce"))
densite2007 = list(pd.to_numeric(etats["Densité 2007"], errors="coerce"))
densite2026 = list(pd.to_numeric(etats["Densité 2026"], errors="coerce"))

# Question 11
print("Question 11")

ordrePop2007 = ordrePopulation(pop2007, etat)
ordrePop2026 = ordrePopulation(pop2026, etat)
ordreDensite2007 = ordrePopulation(densite2007, etat)
ordreDensite2026 = ordrePopulation(densite2026, etat)

# Question 12
print("Question 12")

classementPop = classementPays(ordrePop2007, ordrePop2026)
classementDensite = classementPays(ordreDensite2007, ordreDensite2026)

# Question 13
print("Question 13")

rangPop2007, rangPop2026 = isolerColonnes(classementPop)
rangDensite2007, rangDensite2026 = isolerColonnes(classementDensite)

# Question 14
print("Question 14")

print("Population :")
print("Spearman :", scipy.stats.spearmanr(rangPop2007, rangPop2026))
print("Kendall :", scipy.stats.kendalltau(rangPop2007, rangPop2026))

print("Densité :")
print("Spearman :", scipy.stats.spearmanr(rangDensite2007, rangDensite2026))
print("Kendall :", scipy.stats.kendalltau(rangDensite2007, rangDensite2026))

# Bonus
print("Bonus")

anneesMilitaires = [str(a) for a in range(1949, 2026)]
resMilitaire = comparerToutesLesAnnees(depenses, anneesMilitaires, "", "État")
resMilitaire.to_csv("bonus_depenses_militaires.csv", index=False)
print(resMilitaire.head(10))

anneesPop = [str(a) for a in range(2007, 2027)]
resPop = comparerToutesLesAnnees(etats, anneesPop, "Pop ", "État")
resPop.to_csv("bonus_population.csv", index=False)
print(resPop.head(10))