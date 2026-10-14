"""Função de janela observacional e classificação de harmônicos e aliases."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from lomb_scargle import indices_picos_distintos


def calcular_janela_espectral(
    tempo_relativo: np.ndarray,
    frequencias: np.ndarray,
) -> np.ndarray:
    """Calcular o diagnóstico de janela observacional usado pelo projeto.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.

    Returns
    -------
    numpy.ndarray
        Potências da janela, alinhadas às frequências fornecidas.

    Notes
    -----
    Aplica LombScargle a valores unitários, sem pesos, ajuste de média ou
    centralização, com normalização standard e método auto. Essa definição
    é específica do código; não é o módulo quadrado de uma DFT da amostragem.
    """
    janela = LombScargle(
        tempo_relativo,
        np.ones_like(tempo_relativo),
        fit_mean=False,
        center_data=False,
        normalization="standard",
    )
    return janela.power(frequencias, method="auto")


def analisar_janela_espectral(
    tempo_relativo: np.ndarray,
    frequencias: np.ndarray,
    frequencia_maxima: float,
    resolucao_frequencia: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Montar a grade da janela e identificar até oito picos dominantes.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.
    frequencia_maxima : float
        Limite superior da busca em dia^-1.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.

    Returns
    -------
    tuple of numpy.ndarray
        (frequencias_janela, potencia_janela, indices_janela, frequencias_alias).
        Frequências em dia^-1; índices se referem à grade completa da janela.

    Notes
    -----
    A grade começa em resolucao_frequencia e usa o passo mediano da grade LS.
    O limite de numpy.arange é frequencia_maxima + 0.5*passo.
    """
    passo_frequencia = float(np.median(np.diff(frequencias)))
    frequencia_janela_minima = resolucao_frequencia
    frequencias_janela = np.arange(
        frequencia_janela_minima,
        frequencia_maxima + 0.5 * passo_frequencia,
        passo_frequencia,
    )
    potencia_janela = calcular_janela_espectral(tempo_relativo, frequencias_janela)
    indices_janela = indices_picos_distintos(
        frequencias_janela,
        potencia_janela,
        resolucao_frequencia,
        8,
    )
    frequencias_alias = frequencias_janela[indices_janela]
    return frequencias_janela, potencia_janela, indices_janela, frequencias_alias


def classificar_relacao(
    frequencia: float,
    frequencia_principal: float,
    frequencias_janela: np.ndarray,
    resolucao_frequencia: float,
) -> str:
    """Rotular a proximidade a um período selecionado, harmônico ou alias.

    Parameters
    ----------
    frequencia : float
        Frequência em ciclos por dia.
    frequencia_principal : float
        Frequência do modelo selecionado, em dia^-1.
    frequencias_janela : np.ndarray
        Frequências dos picos da janela usados para procurar aliases, em dia^-1.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.

    Returns
    -------
    str
        "selecionado", rótulo de harmônico/sub-harmônico/possível alias,
        ou "alternativo" quando não há correspondência.

    Notes
    -----
    Usa tolerância <= resolucao_frequencia e retorna a primeira relação
    encontrada na ordem do código. Os rótulos são diagnósticos, não evidência
    probabilística de que um pico seja espúrio.
    """
    if abs(frequencia - frequencia_principal) <= resolucao_frequencia:
        return "selecionado"

    for multiplicador in (2, 3):
        if abs(frequencia - multiplicador * frequencia_principal) <= resolucao_frequencia:
            return f"harmônico {multiplicador}f"
        if abs(multiplicador * frequencia - frequencia_principal) <= resolucao_frequencia:
            return f"sub-harmônico f/{multiplicador}"

    for frequencia_janela in frequencias_janela:
        for multiplicador in (1, 2, 3):
            for ordem_alias in (1, 2):
                for sinal, simbolo in ((1, "+"), (-1, "-")):
                    previsto = abs(
                        multiplicador * frequencia_principal
                        + sinal * ordem_alias * frequencia_janela
                    )
                    if abs(frequencia - previsto) <= resolucao_frequencia:
                        prefixo = "f" if multiplicador == 1 else f"{multiplicador}f"
                        return (
                            f"possível alias |{prefixo}{simbolo}"
                            f"{ordem_alias}×janela|"
                        )
    return "alternativo"
