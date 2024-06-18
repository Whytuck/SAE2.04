import numpy as np
import pandas as pd
import matplotlib as mp 
from sklearn.linear_model import LinearRegression

# Import du fichier avec spécification du délimiteur
VA_CollegeDF = pd.read_csv("/home/etuinfo/jdelouya/Documents/R208/stats_college.txt", delimiter=';')

# Suppression des cases vides
VA_CollegeDF = VA_CollegeDF.dropna()

# Conversion en Array
VA_CollegeAR = VA_CollegeDF.to_numpy()

# Fonction pour centrer-réduire
def centreReduire(T):
    T = np.array(T, dtype=np.float64)
    (n, p) = T.shape
    res = np.zeros((n, p))
    Tmoy = np.mean(T, axis=0)
    Tecart = np.std(T, axis=0)
    for j in range(p):
        res[:, j] = (T[:, j] - Tmoy[j]) / Tecart[j]
    return res

# Centrer-réduire les données
VA_CollegeAR_CR = centreReduire(VA_CollegeAR)

# Création des noms de colonnes pour VA_CollegeDF0
colonne = VA_CollegeDF.columns
ligne = VA_CollegeDF.index

# Création de la DataFrame centrée-réduite
VA_CollegeDF0 = pd.DataFrame(data = VA_CollegeAR_CR, columns = colonne)

# Calcul de la matrice de covariance
MatriceCov = np.cov(VA_CollegeAR_CR, rowvar=False)
print("Matrice de covariance:\n", MatriceCov)

# Sélection des variables explicatives
Y = VA_CollegeAR_CR[:, 0]  # Supposons que la première colonne soit la note moyenne au brevet
X = VA_CollegeAR_CR[:, 1:]  # Les autres colonnes comme variables explicatives

# Régression linéaire multiple
linear_regression = LinearRegression()
linear_regression.fit(X, Y)
a = linear_regression.coef_
print("Coefficients de régression linéaire multiple:", a)

import numpy as np
import matplotlib.pyplot as plt

def DiagBatons(Colonne):
    m = np.min(Colonne) # m contains the minimum value of the column
    M = np.max(Colonne) # M contains the maximum value of the column
    inter = np.linspace(m, M, 21) # list of 21 values ranging from m to M. Use np.linspace function
    plt.figure()
    # plot the histogram for the intervals inter
    plt.hist(Colonne, bins=inter, histtype='bar', align='left', rwidth=0.5)
    plt.show()

# Example usage:
DiagBatons(VA_CollegeAR[:,0])
DiagBatons(VA_CollegeAR[:,1])
DiagBatons(VA_CollegeAR[:,2])
DiagBatons(VA_CollegeAR[:,3])
DiagBatons(VA_CollegeAR[:,4])

