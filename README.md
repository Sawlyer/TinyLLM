# TinyLLM

TinyLLM est un projet personnel dont le but est simple : comprendre ce qui se passe réellement
quand on entraîne un modèle de langage. Le modèle est écrit en PyTorch, sans bibliothèque qui
cache l'architecture Transformer ou la boucle d'entraînement.

Le dépôt contient tout le parcours : téléchargement et préparation des textes, apprentissage d'un
tokenizer, entraînement, sauvegarde des checkpoints, évaluation, génération de texte et mesure de
la vitesse sur une RTX 5080. Le projet sert avant tout à expérimenter et à rendre chaque étape
facile à relire.

La configuration actuelle donne un modèle de 39 854 592 paramètres entraînables. La commande
`tinyllm train --dry-run` recalcule ce nombre directement depuis la configuration.

## Architecture du modèle

TinyLLM est un Transformer autoregressif de type décodeur uniquement. À chaque position, il ne
peut regarder que les tokens précédents. Il apprend donc à prédire le token suivant, exactement
comme lors de la génération.

Les choix actuels sont les suivants :

- vocabulaire BPE de 16 384 tokens ;
- contexte maximal de 512 tokens ;
- vecteurs internes de taille 512 (`d_model`) ;
- 8 blocs Transformer ;
- 8 têtes de requête et 4 têtes clé/valeur (GQA) ;
- MLP SwiGLU avec un rapport d'expansion de 4 ;
- RMSNorm avant chaque sous-couche ;
- positions rotatives RoPE ;
- poids de l'embedding d'entrée partagés avec la tête de sortie ;
- cache clé/valeur utilisé pendant la génération.

Cette configuration représente environ 39,85 M paramètres. Elle reste volontairement assez petite
pour être entraînée et inspectée sur une seule carte graphique.

## Fonctionnalités

- préparation en flux depuis `codelion/fineweb-edu-100M`, séparation déterministe entraînement/
  validation, tokenizer BPE local, identités SHA-256 et publication atomique des artefacts ;
- blocs Transformer pré-normalisés avec RMSNorm, RoPE, attention groupée, SwiGLU, embeddings
  d'entrée et de sortie liés, et un cache KV par couche ;
- entraînement sur un appareil en FP32, FP16 ou BF16, accumulation de gradients, métriques de
  validation, journaux TensorBoard, checkpoints atomiques et reprise déterministe ;
- génération gloutonne ou échantillonnée avec contexte glissant, comparaison de précision et
  exports d'inférence BF16, FP16 ou INT8 vérifiés après rechargement ;
- suite de tests CPU et tests CUDA marqués séparément.

L'architecture et les commandes principales sont décrites directement dans ce README.

## Démarrage rapide Windows avec uv

Il faut Windows 10 ou une version plus récente, Git et Python 3.12. Pour installer `uv`, suivre
le [guide officiel](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
winget install --id=astral-sh.uv -e
git clone <repository-url> TinyLLM
cd TinyLLM
uv sync --extra dev
uv run pytest -m "not cuda"
uv run tinyllm --help
```

Installation classique avec `pip` depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contient les dépendances nécessaires au fonctionnement du projet.
`requirements-dev.txt` ajoute les outils de test et de qualité de code.

Pour utiliser la RTX 5080, installer d'abord les dépendances, puis installer PyTorch CUDA en
dernier avec la commande donnée par [l'installateur officiel](https://pytorch.org/get-started/locally/).
Sinon, `pip` peut choisir une version CPU et remplacer la version CUDA :

```powershell
python -m pip install -r requirements-dev.txt
<commande PyTorch CUDA générée par le site officiel>
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

L'export INT8 est optionnel :

```powershell
python -m pip install torchao==0.18.0
```

`uv sync` suffit pour le développement CPU et les tests. Les roues CUDA dépendent de la version
PyTorch et du pilote installés. Vérifier l'environnement avant l'entraînement :

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

TorchAO est optionnel. Installer l'extra uniquement pour tester l'export INT8 :

```powershell
uv sync --extra dev --extra quantization
```

La documentation TorchAO décrit les possibilités actuelles d'[inférence quantifiée](https://docs.pytorch.org/ao/stable/workflows/inference.html).

## Préparer et inspecter les données

La source par défaut est [`codelion/fineweb-edu-100M`](https://huggingface.co/datasets/codelion/fineweb-edu-100M), un échantillon d'environ 100M tokens provenant de
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
La préparation récupère la révision distante, enregistre son empreinte, entraîne le tokenizer,
écrit les artefacts de tokens en `uint16` et vérifie chaque somme de contrôle avant publication.

```powershell
uv run tinyllm prepare --config configs/tinyllm.yaml
uv run tinyllm inspect-data --config configs/tinyllm.yaml
```

Les sorties ne sont jamais écrasées par défaut. Pour reconstruire volontairement les données,
ajouter `--overwrite`. Pour un test hors ligne, définir `data.dataset_name` vers un fichier texte
UTF-8 local ; chaque ligne non vide devient un document.

La fiche de l'échantillon 100M ne définit pas de licence séparée. La fiche FineWeb-Edu indique
[Open Data Commons Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/). Les textes
web peuvent conserver leurs propres conditions d'utilisation et contenir des données sensibles
ou inexactes ; consulter les fiches sources avant toute redistribution.

### Prévoir l'espace disque

La fiche indique un téléchargement Parquet de 329 Mo. Un corpus de 100M tokens en `uint16` occupe
environ 200 Mo avant la surcharge du système de fichiers. Le cache, les fichiers temporaires, le
tokenizer, les journaux et le corpus peuvent coexister : prévoir au moins 1 Go pour les données.
Un checkpoint avec l'état AdamW occupe plusieurs centaines de Mo ; plusieurs checkpoints et
exports demandent plusieurs Go. Ce sont des estimations de capacité, pas des mesures de débit.

## Entraîner, reprendre, évaluer, générer

Valider la configuration sans allouer l'état d'entraînement CUDA ni démarrer l'optimiseur :

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --dry-run
```

Le mode `dry-run` vérifie l'identité des données, calcule le nombre de paramètres, le batch
effectif, les tokens par étape, la précision demandée et l'appareil cible. Le batch effectif par
défaut est de 16 séquences multipliées par 8 accumulations : 128 séquences, soit 65 536 tokens
par étape.

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --max-steps 20 --checkpoint checkpoints/latest.pt
uv run tinyllm train --config configs/tinyllm.yaml --resume checkpoints/latest.pt --checkpoint checkpoints/resumed.pt --max-steps 10000
uv run tinyllm evaluate --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --batches 8
uv run tinyllm generate --checkpoint checkpoints/resumed.pt --prompt "What is machine learning?"
```

Un checkpoint contient le modèle, l'optimiseur, le scheduler, l'état de précision, les générateurs
aléatoires Python/PyTorch/CUDA, le générateur du loader, la configuration complète et les identités
du modèle, du tokenizer et du corpus. Une reprise refuse toute configuration ou tout artefact
incompatible avant modification. La génération peut récupérer la configuration depuis le
checkpoint ; utiliser `--config` pour valider volontairement une configuration externe.
La configuration par défaut conserve `training.max_steps: 10000` ; la première commande s'arrête
plus tôt sans modifier la configuration enregistrée dans le checkpoint. La reprise reste donc
compatible et continue jusqu'à l'étape 10000.

One-off overrides use validated dotted YAML scalars:

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --set training.max_steps=2000 --set training.warmup_steps=100 --dry-run
```

Les clés inconnues et les valeurs invalides sont refusées. Garder `training.warmup_steps` inférieur
à `training.max_steps`.

## Rapport de précision RTX 5080

Les mesures ci-dessous ont été réalisées le 21 août 2026 sur une RTX 5080, avec PyTorch
2.13.0+cu130, CUDA 13.0, un batch 16 × 512 et le corpus FineWeb-Edu fixe. L'entraînement compare
les mêmes poids initiaux et les mêmes lots sur trois exécutions complètes. L'écart-type reste
inférieur à 1 % du débit moyen pour chaque mode.

| Mode d'entraînement | Débit moyen ± écart-type | VRAM maximale allouée | Loss moyenne |
| --- | ---: | ---: | ---: |
| FP32 | 67,997 ± 316 tokens/s | 5.81 GiB | 9.74017 |
| FP16 | 126,661 ± 988 tokens/s | 4.19 GiB | 9.74015 |
| BF16 | 129,279 ± 1,171 tokens/s | 4.19 GiB | 9.74027 |

BF16 atteint 1,90 fois le débit FP32 avec 28 % de VRAM allouée en moins. FP16 reste à moins de
2,1 % du débit BF16. Toutes les losses mesurées sont finies et l'écart reste inférieur à 0,00012.

L'inférence utilise le checkpoint réel de l'étape 210, huit lots de validation fixes, trois passes
de chauffe puis cinq passes mesurées. Le débit médian est indiqué car la première passe reste un
outlier reproductible de démarrage à froid ; toutes les passes brutes sont conservées.

| Mode d'inférence | Débit médian | VRAM maximale | Perplexité | Écart absolu moyen des logits vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 219,710 tokens/s | 1.70 GiB | 495.704 | 0 |
| FP16 | 491,967 tokens/s | 1.45 GiB | 495.705 | 0.00033 |
| BF16 | 493,492 tokens/s | 1.45 GiB | 495.651 | 0.00266 |
| INT8 | 472,361 tokens/s | 0.85 GiB | 492.364 | 0.00729 |

INT8 réduit de 50 % la mémoire d'inférence allouée par rapport à FP32, avec une loss de validation
similaire sur ce petit checkpoint. Sa perplexité légèrement plus basse ne prouve pas une meilleure
qualité : la comparaison porte sur seulement 65 536 tokens de validation et 210 étapes. L'écart
des logits montre la différence numérique réelle, même quand la loss globale baisse.

Les valeurs détaillées peuvent être régénérées avec les commandes de benchmark ci-dessous.

Pour reproduire les mesures d'entraînement, lancer les trois exécutions puis reconstruire les
résultats d'inférence :

```powershell
uv run --no-sync tinyllm benchmark-precision --config configs/tinyllm.yaml --modes fp32 fp16 bf16 --warmup-steps 20 --measured-steps 100 --output runs/precision-comparison-1.json --overwrite
uv run --no-sync tinyllm benchmark-precision --config configs/tinyllm.yaml --modes fp32 fp16 bf16 --warmup-steps 20 --measured-steps 100 --output runs/precision-comparison-2.json --overwrite
uv run --no-sync tinyllm benchmark-precision --config configs/tinyllm.yaml --modes fp32 fp16 bf16 --warmup-steps 20 --measured-steps 100 --output runs/precision-comparison-3.json --overwrite
uv run --no-sync tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe int8 --output exports/tinyllm-int8.pt --device cuda --overwrite
uv run --no-sync python benchmarks/run_precision_report.py
uv run --no-sync python benchmarks/render_precision_report.py
```

Comparer uniquement des résultats obtenus avec le même logiciel, modèle, corpus, forme de batch,
politique d'alimentation et état thermique. Ces chiffres sont locaux et ne garantissent pas les
performances de toutes les RTX 5080.

## Exports BF16, FP16 et INT8

```powershell
uv run tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe bf16 --output exports/tinyllm-bf16.pt
uv run tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe int8 --output exports/tinyllm-int8.pt
```

Les exports directs BF16/FP16 nécessitent un type d'exécution pris en charge. INT8 utilise les
poids seuls via TorchAO et dépend de la version de bibliothèque, de l'appareil, de la sérialisation
et des noyaux disponibles. L'export n'est accepté qu'après rechargement strict et vérification
numérique déterministe. Une destination existante n'est remplacée qu'avec `--overwrite`.

## Limites

- un seul processus et un seul appareil ; pas d'entraînement distribué ;
- contexte par défaut de 512 tokens et budget de paramètres volontairement réduit ;
- pré-entraînement uniquement : pas d'ajustement aux instructions, aux préférences ou à la
  sécurité, et aucune garantie de factualité ;
- corpus principalement anglophone pouvant contenir biais, doublons, données personnelles ou
  contenu dangereux ;
- échantillon de 100M tokens adapté aux expériences, pas à une qualité généraliste compétitive ;
- support INT8 à vérifier sur le matériel cible ; les tests CPU ne mesurent ni noyaux CUDA ni débit ;
- benchmarks locaux, jamais une garantie universelle.

## Attribution

- échantillon 100M : [fiche codelion/fineweb-edu-100M](https://huggingface.co/datasets/codelion/fineweb-edu-100M) ;
- corpus amont : [fiche HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ;
- article FineWeb : [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) ;
- contexte d'échantillonnage : [The 1 Billion Token Challenge](https://huggingface.co/blog/codelion/optimal-dataset-mixing/) ;
- PyTorch, Hugging Face Datasets, Tokenizers, TensorBoard, Pydantic, PyYAML et TorchAO fournissent
  les outils utilisés par le projet.
