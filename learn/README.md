# Parcours De Cours - Scoring Credit Et Evaluation ML

Ce dossier regroupe les notes importantes sous forme de parcours de cours.
L'objectif est de garder le contenu existant tout en donnant un ordre de
lecture clair.

## Ordre De Lecture

| Module | Fichier | Objectif |
| --- | --- | --- |
| 01 | `01_evaluation_interpretabilite_et_seuils.md` | Comprendre les metriques pour donnees desequilibrees, l'interpretabilite et le choix du seuil de decision. |
| 02 | `02_lift_gain_deciles_scoring.md` | Comprendre les courbes Lift, Gain, Accumulation et les deciles utilises en scoring. |
| 03 | `03_calibration_modeles_scoring_risque.md` | Comprendre pourquoi les probabilites doivent etre calibrees dans un contexte risque/credit. |

## Logique Pedagogique

1. Evaluer correctement un modele avant de parler de performance metier.
2. Lire la valeur business du score avec les courbes lift/gain/deciles.
3. Verifier que les probabilites sont fiables avec la calibration.

## Lien Avec Le Projet

Ces notions correspondent aux etapes aval du pipeline :

- evaluation interne et externe ;
- choix du seuil operationnel ;
- analyse des deciles et du lift ;
- calibration des probabilites ;
- justification du modele devant un comite risque.
