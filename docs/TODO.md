## 24/07/2026 mise en valeur publication

* [ ] analyse des erreurs (?) , pour la science et pour la présentation + solide EMNLP
* [ ] seul annotateur (?)
* [ ] partage du corpus sur HuggingFace + Zenodo (pour respecter règles ANSES)
* [ ] recréation du github avec gitignore corrects (branche main = publique)
* [ ] branche dev avec jean-zay + docs etc (changement gitignore entre les branches)
* [ ] mise a jour du readme (avec tout les liens lors de la publication)
* [ ] passer repo en public
* [ ] posts linkedin, twitter(?)


## 02/07/2026 experiments things to ensure : 

* [x] add newest guidelines to the corpus entities.json and rules.json
* [x] add retry for experiments that did not finish
* [x] fix qwen disable thinking in HF script
* [x] ajouter les règles d'annotations comme un fichier
* [x] ensure that prompt contains all for run_prompting (definitions for all entities, loading and display of rules.json)
* [x] évaluation sur deux versions du corpus : une version macro-entité (pratiques alimentaires et entités climatiques) et une version sous-entités
* [x] Modèles : ajouter Gollie 7B , et des modèles LLM équivalents (qwen 7-8B)
* [x] result table for entity type metrics
* [x] result table with easier dataset (2 entity types)
* [x] add unique mentions in corpus stats
* [x] change prompt figure for separation of system and user part
* [x] ablations (optional)


## 19/06/2026 réunion 

* [x] évaluation 2 métriques : stricte et relax (au moins un overlap et le bon type)
* [x] Ajouter modèle propriétaire état de l'art (OpenRouter) 
* [x] Bien mettre guidelines + exemples (image avec entités discontinues etc) dans l'article


## 04/06/2026 Article ClimateNLP Corpus SF4CD 

* [x] Uniformiser les annotations des différentes version du corpus : DISCONT+NEST / NESTED / FLAT 
* [x] Rédiger guidelines propres à chaque version
* [x] pré annotation avec les sous types d'entités pour les pratiques alimentaires
* [x] dire qu'il s'agit d'une v1 et qu'une v2 viendrait ensuite
* [x] statistiques du corpus : nb doc, nb entités, formes différentes, longueur moyenne des entités, nb phrases, longueur moyenne des phrases, nb entités par doc en moyenne, 
* [x] performance des baselines : gliner, gollie, llm (petits ? moyens ?)
* [x] evaluation des méthodes : stricte, compter les entités discontinues comme fausses // 3 évaluations (flat, nested, discontinuous)
* [x] impact climatique (consommation CO2)
* [x] discussion : entités discontinues non traitées, ouvrir la dessus
* [x] comment le corpus est construit 
 
