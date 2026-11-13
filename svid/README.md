# Vérification de SVID sur Qwen2.5-0.5B

Une expérience annexe au [projet principal](../README.md), menée après la lecture de l'article de Ba-Hien Tran et Van Minh Nguyen, [*Highly Efficient and Effective LLMs with Multi-Boolean Architectures*](https://arxiv.org/abs/2505.22811) (ICLR 2026). Il s'agit d'une vérification de leur idée sur mon modèle, pas d'une contribution nouvelle.

## L'idée vérifiée

Dans le projet principal, approximer les poids de Qwen par une SVD tronquée détruit le modèle, car leurs spectres sont presque aussi plats que ceux d'une matrice aléatoire.

L'article initialise ses poids booléens avec SVID, une décomposition reprise de OneBit (Xu et al., 2024). Elle garde le signe de chaque poids et n'approxime que leurs tailles |W| par un rang 1 :

$$W \approx \mathrm{sign}(W) \odot \left(s_{\text{out}} \, s_{\text{in}}^T\right)$$

où ⊙ est le produit élément par élément et s_out s_inᵀ le rang 1 de |W| obtenu par SVD. Avec plusieurs noyaux, chaque nouveau noyau approxime le résidu laissé par les précédents. Un noyau coûte environ 1 bit par poids : un bit de signe, plus deux vecteurs d'échelles.

**Question :** sur Qwen, les tailles des poids sont-elles assez proches d'un rang 1 pour que SVID réussisse là où la SVD tronquée échoue ?

## Méthode

Pour chacune des 168 matrices compressées dans le projet principal (`q`, `k`, `v`, `o`, `gate`, `up` et `down` de chaque bloc), on mesure l'erreur relative ‖W − Ŵ‖ / ‖W‖ de :

- un rang 1 direct de W ;
- SVID avec 1, 2 et 3 noyaux ;
- une SVD tronquée de W qui utilise autant de bits qu'un noyau SVID.

On fait les mêmes mesures sur une matrice gaussienne de même forme pour chaque type de couche, qui sert de référence « sans structure ». Seuls les poids sont lus : le modèle n'est ni réentraîné, ni évalué.

## Résultats

Erreur relative moyenne sur les 24 blocs ([results.csv](results.csv)) :

| Matrice | Rang 1 direct | SVID, 1 noyau | 2 noyaux | 3 noyaux | SVD au même nombre de bits qu'un noyau |
| --- | --- | --- | --- | --- | --- |
| `q_proj` | 0,982 | 0,616 | 0,382 | 0,268 | 0,819 |
| `k_proj` | 0,976 | 0,618 | 0,380 | 0,263 | 0,861 |
| `v_proj` | 0,990 | 0,619 | 0,380 | 0,264 | 0,935 |
| `o_proj` | 0,991 | 0,625 | 0,393 | 0,281 | 0,874 |
| `gate_proj` | 0,992 | 0,616 | 0,382 | 0,268 | 0,908 |
| `up_proj` | 0,996 | 0,615 | 0,380 | 0,266 | 0,936 |
| `down_proj` | 0,995 | 0,616 | 0,383 | 0,271 | 0,921 |
| *gaussienne* | *0,993 à 0,999* | *0,60* | *0,36* | *0,24* | *0,94 à 0,95* |

## Ce qu'on observe

- **SVID fait mieux qu'un rang 1 direct** (0,62 contre 0,99 en moyenne), comme l'énonce l'article (Proposition 4.1). Ce n'est pas une comparaison à taille égale : dans l'article, les poids sont booléens, donc les signes font partie du modèle, et la question est seulement de choisir les échelles. Ici, les signes coûtent environ 1 bit par poids, alors que le rang 1 direct n'en coûte qu'environ 0,02 en moyenne.
- **À nombre de bits égal, SVID fait mieux qu'une SVD tronquée** : 0,62 contre 0,91 en moyenne pondérée par le nombre de poids. Avec le budget d'un noyau, la SVD tronquée garde environ 29 valeurs singulières pour une matrice 896 × 896, mais l'erreur reste élevée. C'est la comparaison qui compte du point de vue de la compression : à budget égal, il vaut mieux dépenser ses bits en signes qu'en valeurs singulières supplémentaires.
- **Ce gain ne vient pas d'une structure propre à Qwen.** Sur une matrice gaussienne, SVID fait même légèrement mieux (0,60). La raison : |W| a une moyenne positive, et cette composante constante, qui est de rang 1, porte à elle seule une grande part de l'énergie de |W| (environ 2/π pour une gaussienne). Le rang 1 de |W| capture donc surtout la taille moyenne des poids.
- **Les tailles des poids de Qwen ne sont donc pas proches d'un rang 1** : le rang 1 de |W| laisse environ 38 % de son énergie dans le résidu, comme pour une matrice aléatoire.
- **Chaque noyau ajouté multiplie l'erreur par environ 0,62** (0,62, puis 0,38, puis 0,27), au même rythme que sur la gaussienne.

## Limites

- On mesure l'erreur sur les poids, pas la perplexité : une erreur faible sur les poids ne garantit pas un modèle utilisable.
- On ne teste que l'initialisation sans données. L'article réentraîne ensuite ses noyaux, ce qui n'est pas fait ici.
- Un seul modèle, de petite taille.

## Reproduire

Depuis la racine du dépôt (CPU, quelques minutes) :

```bash
python -m svid.svid_check
```
