"""Lomb–Scargle ponderado, grade de frequências e picos distintos."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


def criar_lomb_scargle(
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
) -> LombScargle:
    """Construir o Lomb–Scargle ponderado com média flutuante e um harmônico.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).

    Returns
    -------
    astropy.timeseries.LombScargle
        Objeto configurado com dy=erro, fit_mean=True, center_data=True,
        nterms=1 e normalization="standard".

    Notes
    -----
    A criação do objeto não calcula a grade nem o periodograma.
    """
    ls = LombScargle(
        tempo_relativo,
        magnitude,
        dy=erro,
        fit_mean=True,
        center_data=True,
        nterms=1,
        normalization="standard",
    )
    return ls


def calcular_periodograma(
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    frequencia_minima: float,
    frequencia_maxima: float,
    amostras_por_pico: int,
) -> tuple[LombScargle, np.ndarray, np.ndarray]:
    """Calcular a grade automática e a potência Lomb–Scargle ponderada.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    frequencia_minima : float
        Limite inferior da busca em dia^-1.
    frequencia_maxima : float
        Limite superior da busca em dia^-1.
    amostras_por_pico : int
        Sobreamostragem da grade em relação à escala 1/T.

    Returns
    -------
    tuple
        (ls, frequencias, potencia): objeto LombScargle e dois vetores
        alinhados com frequência em dia^-1 e potência adimensional.

    Notes
    -----
    Usa autofrequency com os limites informados e power(method="auto").
    O último ponto segue o arredondamento da grade do Astropy.
    Lança RuntimeError se alguma potência calculada não for finita.
    """
    ls = criar_lomb_scargle(tempo_relativo, magnitude, erro)
    frequencias = ls.autofrequency(
        samples_per_peak=amostras_por_pico,
        minimum_frequency=frequencia_minima,
        maximum_frequency=frequencia_maxima,
    )
    potencia = ls.power(frequencias, method="auto")

    if not np.all(np.isfinite(potencia)):
        raise RuntimeError("O periodograma produziu potências não finitas.")
    return ls, frequencias, potencia


def indices_picos_distintos(
    frequencias: np.ndarray,
    potencia: np.ndarray,
    resolucao_frequencia: float,
    quantidade: int,
) -> np.ndarray:
    """Selecionar máximos locais, incluindo os extremos e o máximo global.

    Parameters
    ----------
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.
    potencia : np.ndarray
        Potências alinhadas à grade, do sinal ou da janela.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.
    quantidade : int
        Número máximo de índices retornados.

    Returns
    -------
    numpy.ndarray
        Índices inteiros de até quantidade picos, por potência decrescente.

    Notes
    -----
    A distância de find_peaks é ceil(resolucao_frequencia/passo), no mínimo
    1. Os extremos e o máximo global são acrescentados depois, sem reaplicar
    a distância mínima. Lança ValueError para tamanhos diferentes ou < 2.
    """
    if len(frequencias) != len(potencia) or len(frequencias) < 2:
        raise ValueError("A grade e a potência devem ter o mesmo tamanho, maior que 1.")
    passo = float(np.median(np.diff(frequencias)))
    distancia = max(1, int(np.ceil(resolucao_frequencia / passo)))
    indices, _ = find_peaks(potencia, distance=distancia)

    # ``find_peaks`` não considera as extremidades. Elas precisam entrar se
    # forem máximos locais, pois um sinal pode estar junto ao limite pesquisado.
    extremos: list[int] = []
    if potencia[0] >= potencia[1]:
        extremos.append(0)
    if potencia[-1] >= potencia[-2]:
        extremos.append(len(potencia) - 1)
    indice_maximo = int(np.nanargmax(potencia))
    indices = np.unique(np.append(indices, [indice_maximo, *extremos])).astype(int)
    if len(indices) == 0:
        return np.array([indice_maximo], dtype=int)

    ordem = np.argsort(potencia[indices])[::-1]
    return indices[ordem[:quantidade]]
