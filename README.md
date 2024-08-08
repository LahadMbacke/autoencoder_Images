# Autoencodeur pour FashionMNIST

Ce dépôt contient une implémentation en PyTorch d'un modèle autoencodeur destiné à reconstruire des images du jeu de données FashionMNIST. L'autoencodeur apprend à compresser puis décompresser les images, apprenant ainsi une représentation de données de dimension inférieure.

## Architecture du modèle

L'autoencodeur est composé des éléments suivants :

* **Encodeur** :
 * Couche linéaire : `784` -> `128`
 * Activation ReLU
 * Couche linéaire : `128` -> `64` 
 * Activation ReLU
 * Couche linéaire : `64` -> `9`
* **Décodeur** :
 * Couche linéaire : `9` -> `64`
 * Activation ReLU
 * Couche linéaire : `64` -> `128`
 * Activation ReLU
 * Couche linéaire : `128` -> `784` 
 * Activation Sigmoid

L'encodeur compresse l'entrée de 784 dimensions à 9 dimensions, et le décodeur reconstruit les données de 9 dimensions à 784 dimensions.

## Entraînement

Le modèle est entraîné avec la fonction de perte MSE (Mean Squared Error) et l'optimiseur Adam avec un taux d'apprentissage de 1e-3. L'entraînement est effectué pendant 100 époques, avec l'affichage de la perte toutes les 100 itérations.

## Visualisation

Après l'entraînement, le script visualise les images originales et reconstruites à partir du jeu de test. Il sélectionne le premier lot d'images, les fait passer dans le modèle autoencodeur, et affiche côte à côte les images originales et reconstruites.

## Dépendances

- Python 3.7+
- PyTorch 1.7+
- Matplotlib
- Torchvision

Vous pouvez installer les packages requis avec pip :
Ce projet a été inspiré par la documentation https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html et les tutoriels PyTorch, ainsi que par divers articles et billets de blog sur les autoencodeurs et l'apprentissage non supervisé de caractéristiques.