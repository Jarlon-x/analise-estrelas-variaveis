"""Refinamento de frequências e seleção dos candidatos por BIC."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import minimize_scalar

from fourier import AjusteFourier, avaliar_fourier
from lomb_scargle import indices_picos_distintos


def refinar_frequencia(
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    frequencia_inicial: float,
    frequencia_minima: float,
    frequencia_maxima: float,
    resolucao_frequencia: float,
    n_termos_maximo: int,
    ordem_fixa: bool,
) -> AjusteFourier | None:
    """Refinar uma semente de frequência e escolher a ordem de menor BIC.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    frequencia_inicial : float
        Semente positiva em dia^-1, originada de um pico ou sub-harmônico.
    frequencia_minima : float
        Limite inferior da busca em dia^-1.
    frequencia_maxima : float
        Limite superior da busca em dia^-1.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.
    n_termos_maximo : int
        Maior ordem de Fourier examinada; torna-se a ordem usada se fixa.
    ordem_fixa : bool
        Se True, usa somente a ordem máxima; caso contrário, seleciona por BIC.

    Returns
    -------
    AjusteFourier or None
        Melhor ajuste válido entre as ordens examinadas, ou None se todas falharem.

    Notes
    -----
    Busca em +/- 0.5*resolucao_frequencia, respeitando os limites fornecidos.
    Cada ordem usa 25 pontos locais e minimize_scalar(method="bounded")
    entre vizinhos do melhor ponto; xatol=max(resolucao/10000, 1e-12).
    Erros ValueError, LinAlgError e FloatingPointError dos ajustes são tratados.
    """
    meia_largura = 0.5 * resolucao_frequencia
    limite_inferior = max(frequencia_minima, frequencia_inicial - meia_largura)
    limite_superior = min(frequencia_maxima, frequencia_inicial + meia_largura)

    ordens = [n_termos_maximo] if ordem_fixa else range(1, n_termos_maximo + 1)
    ajustes: list[AjusteFourier] = []

    for n_termos in ordens:
        def objetivo(frequencia: float) -> float:
            """Avaliar a função objetivo do refinamento para a ordem corrente.

            Parameters
            ----------
            frequencia : float
                Frequência em ciclos por dia.

            Returns
            -------
            float
                Chi-quadrado do ajuste; infinito se ocorrer um erro numérico tratado.

            Notes
            -----
            Captura tempos, magnitudes, erros e n_termos da função externa.
            """
            try:
                return avaliar_fourier(
                    tempo_relativo,
                    magnitude,
                    erro,
                    frequencia,
                    n_termos,
                ).chi2
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return np.inf

        if limite_superior <= limite_inferior:
            frequencia = frequencia_inicial
        else:
            # Uma busca curta em grade evita aplicar o método limitado a um
            # intervalo que contenha mais de um mínimo local.
            grade_local = np.linspace(limite_inferior, limite_superior, 25)
            valores = np.asarray([objetivo(f) for f in grade_local])
            if not np.any(np.isfinite(valores)):
                continue
            indice = int(np.nanargmin(valores))
            esquerda = grade_local[max(0, indice - 1)]
            direita = grade_local[min(len(grade_local) - 1, indice + 1)]
            frequencia = float(grade_local[indice])
            if direita > esquerda:
                resultado = minimize_scalar(
                    objetivo,
                    bounds=(float(esquerda), float(direita)),
                    method="bounded",
                    options={"xatol": max(resolucao_frequencia / 10000.0, 1e-12)},
                )
                if resultado.success and np.isfinite(resultado.fun):
                    frequencia = float(resultado.x)

        try:
            ajustes.append(
                avaliar_fourier(
                    tempo_relativo,
                    magnitude,
                    erro,
                    frequencia,
                    n_termos,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

    return min(ajustes, key=lambda ajuste: ajuste.bic) if ajustes else None


def selecionar_candidatos(
    ls: LombScargle,
    frequencias: np.ndarray,
    potencia: np.ndarray,
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    resolucao_frequencia: float,
    n_candidatos: int,
    n_termos_maximo: int,
    ordem_fixa: bool,
    incluir_subharmonicos: bool = True,
) -> tuple[list[AjusteFourier], list[str]]:
    """Refinar picos e sub-harmônicos e ordenar os modelos candidatos.

    Parameters
    ----------
    ls : LombScargle
        Objeto LombScargle ponderado construído para os dados analisados.
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.
    potencia : np.ndarray
        Potência Lomb–Scargle adimensional, alinhada à grade de frequências.
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.
    n_candidatos : int
        Número máximo de picos LS usados como sementes antes dos sub-harmônicos.
    n_termos_maximo : int
        Maior ordem de Fourier examinada; torna-se a ordem usada se fixa.
    ordem_fixa : bool
        Se True, usa somente a ordem máxima; caso contrário, seleciona por BIC.
    incluir_subharmonicos : bool, default=True
        Se True, adiciona f/2 e f/3 dos até cinco maiores picos, dentro da grade.

    Returns
    -------
    tuple of lists
        (ajustes, origens), listas alinhadas. O primeiro ajuste é o selecionado;
        os demais seguem BIC crescente. Retorna ([], []) se nenhum for válido.

    Notes
    -----
    Une soluções separadas por menos de 0.5*resolucao_frequencia, mantendo
    o menor BIC. Entre candidatos com BIC <= BIC_min + 2, prefere maior
    potência LS e depois menor ordem. n_candidatos limita as sementes LS,
    não o tamanho final das listas após acrescentar sub-harmônicos.
    """
    indices = indices_picos_distintos(
        frequencias,
        potencia,
        resolucao_frequencia,
        n_candidatos,
    )
    sementes: list[tuple[float, str]] = [
        (float(frequencias[indice]), "pico LS") for indice in indices
    ]

    if incluir_subharmonicos:
        for indice in indices[: min(5, len(indices))]:
            frequencia = float(frequencias[indice])
            for divisor in (2, 3):
                subharmonico = frequencia / divisor
                if frequencias[0] <= subharmonico <= frequencias[-1]:
                    sementes.append((subharmonico, f"pico LS/{divisor}"))

    sementes_ordenadas = sorted(
        sementes,
        key=lambda item: float(ls.power(item[0])),
        reverse=True,
    )
    ajustes: list[AjusteFourier] = []
    origens: list[str] = []
    for frequencia, origem in sementes_ordenadas:
        ajuste = refinar_frequencia(
            tempo_relativo,
            magnitude,
            erro,
            frequencia,
            float(frequencias[0]),
            float(frequencias[-1]),
            resolucao_frequencia,
            n_termos_maximo,
            ordem_fixa,
        )
        if ajuste is None:
            continue
        duplicados = [
            indice
            for indice, existente in enumerate(ajustes)
            if abs(ajuste.frequencia - existente.frequencia)
            < 0.5 * resolucao_frequencia
        ]
        if duplicados:
            indice = duplicados[0]
            if ajuste.bic < ajustes[indice].bic:
                ajustes[indice] = ajuste
                origens[indice] = origem
            continue
        ajustes.append(ajuste)
        origens.append(origem)

    if not ajustes:
        return [], []

    # Se dois modelos têm ΔBIC < 2, os dados não os distinguem de forma
    # relevante. Nesse conjunto, o pico sinusoidal mais forte funciona como
    # desempate e impede que f/2 ou f/3 vença apenas por ruído numérico.
    bic_minimo = min(ajuste.bic for ajuste in ajustes)
    competitivos = [
        indice
        for indice, ajuste in enumerate(ajustes)
        if ajuste.bic <= bic_minimo + 2.0
    ]
    indice_escolhido = max(
        competitivos,
        key=lambda indice: (
            float(ls.power(ajustes[indice].frequencia)),
            -ajustes[indice].n_termos,
        ),
    )
    restantes = sorted(
        (indice for indice in range(len(ajustes)) if indice != indice_escolhido),
        key=lambda indice: ajustes[indice].bic,
    )
    ordem = [indice_escolhido, *restantes]
    ajustes = [ajustes[indice] for indice in ordem]
    origens = [origens[indice] for indice in ordem]
    return ajustes, origens
