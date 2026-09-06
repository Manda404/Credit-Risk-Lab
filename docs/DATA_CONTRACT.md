# Contrat de données et de cible

Ce dépôt utilise un dataset synthétique sans date de décision ni identifiant
d'emprunteur. Il ne peut donc démontrer ni validation hors temps, ni absence de
chevauchement d'emprunteurs. Le mode `random_experimental` est réservé à la
démonstration locale.

## Cible obligatoire pour une utilisation réelle

Avant tout entraînement, le propriétaire des données doit renseigner et faire
approuver : événement de défaut, horizon de performance, jours d'impayés,
population admissible, exclusions, date de décision, date de maturité du label,
et traitement des prêts encore indéterminés. Dans ce laboratoire, `loan_status=1`
est traité comme la classe de risque positive, sans prétendre qu'il s'agit d'une
définition réglementaire de défaut.

## Disponibilité des variables

Chaque variable doit avoir un timestamp `available_at` antérieur ou égal à la
date du scoring. `loan_int_rate`, `loan_percent_income` et leurs dérivés sont
conditionnels : ils ne sont admissibles que si le pricing est effectivement
connu avant la décision évaluée. Sans preuve de lineage temporel, ils doivent
être exclus.

## Colonnes de production minimales

- `application_id` unique et immuable ;
- `borrower_id` pseudonymisé pour empêcher les croisements de splits ;
- `decision_date` horodatée en UTC ;
- features avec types, unités, plages et politique de valeurs manquantes ;
- label et date de maturité du label, stockés séparément des entrées de scoring.
