"""Quantification symétrique des poids (écrite par Adrien)."""
import torch


def quantize(w: torch.Tensor, bits: int, granularity) -> torch.Tensor:
    """Quantifie puis déquantifie une matrice de poids (compression simulée).

    Args:
        w: matrice de poids (m × n), en général en fp16.
        bits: nombre de bits b, parmi 8, 4, 3 ou 2.
        granularity: portée de chaque échelle :
            - "tensor" : une seule échelle pour toute la matrice ;
            - "row" : une échelle par ligne ;
            - un entier g : une échelle par groupe de g poids consécutifs
              dans une ligne (n doit être divisible par g).

    Méthode (symétrique, calculs en float32) :
        q_max = 2^(b-1) - 1
        s     = max|w_groupe| / q_max
        ŵ     = s · clamp(round(w / s), -q_max - 1, q_max)

    Une échelle nulle (groupe entièrement à zéro) ne doit pas produire de NaN.
    La matrice w d'entrée ne doit pas être modifiée.

    Returns:
        ŵ, de même forme et de même type (dtype) que w.
    """

    m, n = w.shape 
    x = w.float() # on passe en float32 pour avoir de meilleurs arrondis
    q_max = 2**(bits - 1) - 1

    if granularity == "tensor":
        x = x.reshape(1,m*n)

    elif granularity == "row":
        # rien : les lignes de la matrice sont déjà les groupes
        pass
    else:
        x = x.reshape(m, n // granularity, granularity)

    
    s = x.abs().amax(dim=-1, keepdim=True) / q_max
    s = torch.where(s == 0, 1.0, s) # si s = 0 on remplace par 1 pour ne pas diviser par 0
        
    quant_x = torch.clamp(torch.round(x / s), -q_max - 1, q_max) * s

    quant_x = quant_x.reshape(m,n)
    quant_x = quant_x.to(w.dtype)

    return quant_x
