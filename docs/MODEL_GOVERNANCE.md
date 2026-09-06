# Gouvernance du modèle

Le modèle n'est promouvable que si le candidat a été choisi sur validation,
évalué une seule fois sur un test hors temps, calibré, comparé à la baseline
logistique, et approuvé indépendamment. Le seuil doit optimiser une fonction de
coût documentée ; Youden J n'est qu'un exemple pédagogique.

L'artefact doit inclure versions runtime, schéma, hash des données, commit Git,
configuration, définition de cible, métriques de validation/test et résultats
d'équité. MLflow sert au tracking et au registre ; la promotion, le rollback et
les droits d'approbation doivent être configurés sur une infrastructure partagée.

Le monitoring couvre schéma, fraîcheur, catégories nouvelles, distributions,
scores, décisions, performance différée, calibration et groupes sensibles. Un
drift seul déclenche une investigation, jamais un réentraînement automatique.
