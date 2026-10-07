#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

# C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
# Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
# Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

# Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1 : Théorie de l'échantillonnage (intervalles de fluctuation)
# L'échantillonnage se base sur la répétitivité.
print("Question 1")
print("Résultat sur le calcul d'un intervalle de fluctuation")
echantillon = ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv")
moyenne_pour = round(echantillon["Pour"].mean())
moyenne_contre = round(echantillon["Contre"].mean())
moyenne_sans_opinion = round(echantillon["Sans opinion"].mean())

print("Moyenne Pour :", moyenne_pour)
print("Moyenne Contre :", moyenne_contre)
print("Moyenne Sans opinion :", moyenne_sans_opinion)

somme_moyennes = moyenne_pour + moyenne_contre + moyenne_sans_opinion

frequence_pour = round(moyenne_pour / somme_moyennes, 2)
frequence_contre = round(moyenne_contre / somme_moyennes, 2)
frequence_sans_opinion = round(moyenne_sans_opinion / somme_moyennes, 2)

print("Fréquence Pour :", frequence_pour)
print("Fréquence Contre :", frequence_contre)
print("Fréquence Sans opinion :", frequence_sans_opinion)

pour_population = 852
contre_population = 911
sans_opinion_population = 422
somme_population = pour_population + contre_population + sans_opinion_population
frequence_pour_population = round(pour_population / somme_population, 2)
frequence_contre_population = round(contre_population / somme_population, 2)
frequence_sans_opinion_population = round(sans_opinion_population / somme_population, 2)
print("Fréquence Pour population :", frequence_pour_population)
print("Fréquence Contre population :", frequence_contre_population)
print("Fréquence Sans opinion population :", frequence_sans_opinion_population)
zC = 1.96

marge_pour = zC * math.sqrt(frequence_pour * (1 - frequence_pour) / 1000)
marge_contre = zC * math.sqrt(frequence_contre * (1 - frequence_contre) / 1000)
marge_sans_opinion = zC * math.sqrt(frequence_sans_opinion * (1 - frequence_sans_opinion) / 1000)

print("Intervalle de fluctuation Pour :", frequence_pour - marge_pour, "-", frequence_pour + marge_pour)
print("Intervalle de fluctuation Contre :", frequence_contre - marge_contre, "-", frequence_contre + marge_contre)
print("Intervalle de fluctuation Sans opinion :", frequence_sans_opinion - marge_sans_opinion, "-", frequence_sans_opinion + marge_sans_opinion)


## Question 2 : Théorie de l'estimation (intervalles de confiance)
# L'estimation se base sur l'effectif.

print("Question 2")
print("Résultat sur le calcul d'un intervalle de confiance")

premier_echantillon = list(echantillon.iloc[0])

taille_echantillon = sum(premier_echantillon)

print("Taille du premier échantillon :", taille_echantillon)

frequence_pour_q2 = round(premier_echantillon[0] / taille_echantillon, 2)
frequence_contre_q2 = round(premier_echantillon[1] / taille_echantillon, 2)
frequence_sans_opinion_q2 = round(premier_echantillon[2] / taille_echantillon, 2)

print("Fréquence Pour :", frequence_pour_q2)
print("Fréquence Contre :", frequence_contre_q2)
print("Fréquence Sans opinion :", frequence_sans_opinion_q2)

marge = 1 / math.sqrt(taille_echantillon)

print("Marge :", marge)

frequence_pour_basse = frequence_pour_q2 - marge
frequence_pour_haute = frequence_pour_q2 + marge

frequence_contre_basse = frequence_contre_q2 - marge
frequence_contre_haute = frequence_contre_q2 + marge

frequence_sans_opinion_basse = frequence_sans_opinion_q2 - marge
frequence_sans_opinion_haute = frequence_sans_opinion_q2 + marge

print("Intervalle de confiance Pour :", round(frequence_pour_basse, 2), "-", round(frequence_pour_haute, 2))
print("Intervalle de confiance Contre :", round(frequence_contre_basse, 2), "-", round(frequence_contre_haute, 2))
print("Intervalle de confiance Sans opinion :", round(frequence_sans_opinion_basse, 2), "-", round(frequence_sans_opinion_haute, 2))


## Question 3 : Théorie de la décision
# La décision se base sur la notion de risques alpha et bêta.

print("Question 3")
print("Théorie de la décision")

loi1 = ouvrirUnFichier("data/Loi-normale-Test-1.csv")
loi2 = ouvrirUnFichier("data/Loi-normale-Test-2.csv")

test1 = scipy.stats.shapiro(loi1.iloc[:, 0])
test2 = scipy.stats.shapiro(loi2.iloc[:, 0])

print("Test de Shapiro - Loi 1 :", test1)
print("Test de Shapiro - Loi 2 :", test2)

# Question bonus
print("La Loi 2 suit une loi géométrique.")