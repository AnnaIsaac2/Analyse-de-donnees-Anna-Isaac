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
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
                    
    return classement

    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element][1]])
    return classement

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
plt.ylabel("Dépenses militaires")
plt.savefig("depenses2025.png")
plt.close()

# Question 6
print("Question 6")

logDepenses2025 = conversionLog(depenses2025SansNaN)

# Question 7
print("Question 7")
#Un test sur les rangs est possible car les dépenses militaires peuvent être classées selon un ordre de grandeur.

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

rang2007 = []
rang2026 = []

for element in classementPop:
    rang2007.append(element[0])
    rang2026.append(element[1])

# Question 14
print("Question 14")

spearman = scipy.stats.spearmanr(rang2007, rang2026)
kendall = scipy.stats.kendalltau(rang2007, rang2026)

print("Spearman :", spearman)
print("Kendall :", kendall)

# Bonus
print("Bonus")

annees = ["2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

classementsMilitaires = []

for annee in annees:
    depensesAnnee = list(pd.to_numeric(depenses[annee], errors="coerce"))
    classementAnnee = ordrePopulation(depensesAnnee, list(depenses["État"]))
    classementsMilitaires.append(classementAnnee)

for i in range(len(annees)):
    for j in range(i + 1, len(annees)):
        classement = classementPays(classementsMilitaires[i], classementsMilitaires[j])

        rang1 = []
        rang2 = []

        for element in classement:
            rang1.append(element[0])
            rang2.append(element[1])

        spearman = scipy.stats.spearmanr(rang1, rang2)
        kendall = scipy.stats.kendalltau(rang1, rang2)

        print(annees[i], "-", annees[j])
        print("Spearman :", spearman.correlation)
        print("Kendall :", kendall.correlation)