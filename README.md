# Rapport de Recherche : Développement d'un LLM "From Scratch"

## Introduction & Contexte
Ce projet se concentre sur le développement d'un LLM (Large Language Model) entièrement à partir de zéro en utilisant PyTorch. Plutôt que d'utiliser des APIs existantes ou des modèles pré-entraînés sans savoir comment ils fonctionnent, je voulais comprendre les mécanismes qui régissent ces systèmes : de la tokenisation à la génération de texte, en passant par les mécanismes d'attention.

Il ne s'agit pas ici de concurrencer Mistral ou Claude, mais de reproduire le fonctionnement des LLMs. L'objectif était d'implémenter un modèle de type Transformer entièrement à partir de zéro pour démystifier cette architecture.

## Architecture Technique
Ce modèle est un **Decoder-only Transformer**, inspiré de l'architecture GPT.

### 1. Tokenisation
J'ai utilisé **SentencePiece** pour la tokenisation. Contrairement à une tokenisation simple par mot ou caractère, cela permet de gérer efficacement un vocabulaire de taille fixe tout en couvrant des mots rares via des sous-unités lexicales.
- **Taille du vocabulaire** : 3000 tokens, volontairement restreint pour l'expérimentation.

### 2. Le Modèle
L'architecture, définie dans `src/model_transformer.py`, est le fruit de l'étude combinée du papier **"Attention Is All You Need"** (pour la théorie et les formules mathématiques) et des travaux d'**Andrej Karpathy** (pour la mise en pratique). C'est un **Decoder-only Transformer**, adapté pour la génération de texte :

*   **Embeddings** : Une couche vectorielle qui transforme chaque token en vecteur dense de dimension `384`.
*   **Positional Encoding** : J'ai implémenté un encodage positionnel sinusoïdal.
*   **Blocs Transformer (8 couches)** :
    *   **Multi-Head Attention** : Le cœur du système avec 8 têtes d'attention.
    *   **Feed-Forward Networks** : Des réseaux de neurones denses classiques (MLP).

### 3. Hyperparamètres Clés
*   **Contexte (Block Size)** : 256 tokens.
*   **Paramètres** : ~15 millions car 12 x 384 x 384 x 8 + 3000 x 384.
*   **Optimiseur** : AdamW avec un learning rate de `3e-4`.

## La Boucle d'Entraînement (Training Loop)
J'ai implémenté une boucle d'entraînement complète en PyTorch (`src/train.py`) qui :
1.  Charge les données tokenisées par batchs.
2.  Calcule la **Cross-Entropy Loss** entre la prédiction et le token réel suivant.
3.  Backpropagate l'erreur et met à jour les poids.
4.  Gère les checkpoints pour sauvegarder la progression.

*Note technique : Le code est optimisé pour tourner sur Apple Silicon via `mps`.*

## Apprentissages & Limitations
C'est ici que la réalité du Deep Learning moderne m'a rattrapé.

### Ce que j'ai appris
*   **La logique mathématique** : Comprendre la logique mathématique derrière les formule dans le document "Attention is All You Need" permet de vraiment comprendre comment le modèle fonctionne.
*   **La puissance de l'Attention** : Implémenter `scaled_dot_product_attention` à la main permet de vraiment comprendre comment le modèle "requête" (Query) son contexte (Key) pour en extraire de la valeur (Value).
*   **La complexité de l'infrastructure** : Gérer le chargement des données, le GPU (MPS), et la sauvegarde demande autant de travail que le modèle lui-même.

### Les Murs rencontrés
*   **Ressources de Calcul** : Entraîner un LLM, même petit, "from scratch" sur des données financières demande une puissance de grande puissance de calcul. Avec mon setup local, la perte (loss) descend, mais beaucoup trop lentement pour atteindre une cohérence linguistique.
*   **Volume de Données** : Un modèle language a besoin de gigaoctets de texte pour ne serait-ce que commencer à générer des phrases cohérentes et la création d'un corpus financier en français est un projet majeur qui demande beaucoup de temps et de ressources.

## Pivot : Fine-Tuning

Plutôt que de continuer à essayer de créer un LLM "from scratch", la prochaine étape logique est le **Fine-Tuning**.
Je vais basculer vers l'utilisation d'un modèle open-source robuste comme **Mistral 7B** ou **Llama 3**. L'idée est de :
1.  Prendre un modèle pré-entraîné
2.  Lui apprendre les termes et les spécificités de la finance via des techniques comme **LoRA** ou **QLoRA**.

Ce premier projet reste ma pierre angulaire : je sais maintenant exactement ce qui se passe quand j'appelle `model.generate()`.

## Ressources & Inspirations
Pour mener à bien ce projet, je me suis appuyé sur deux ressources principales :
*   **"Attention Is All You Need"** : Le papier de recherche fondateur pour la compréhension théorique et les formules mathématiques de l'architecture Transformer.
*   **Andrej Karpathy** : Ses vidéos pédagogiques (notamment sur la construction de GPT 'from scratch') ont été cruciales pour la mise en pratique.