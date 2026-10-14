"""Coordenação das etapas de análise e montagem do resultado completo."""

from __future__ import annotations

import numpy as np

from estatistica import (
    binning_ponderado,
    bootstrap_periodos,
    calcular_significancia,
    resumir_bootstrap,
)
from fourier import matriz_fourier, parametros_fourier
from janela_espectral import analisar_janela_espectral
from lomb_scargle import calcular_periodograma
from resultados import criar_tabela_candidatos, criar_tabela_janela
from selecao_modelos import selecionar_candidatos


def analisar_serie(
    tempo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    periodo_minimo: float = 0.1,
    periodo_maximo: float = 100.0,
    n_termos: int = 4,
    amostras_por_pico: int = 10,
    n_candidatos: int = 15,
    n_bins: int = 50,
    n_bootstrap: int = 200,
    semente: int = 12345,
    ordem_fixa: bool = False,
    metodo_bootstrap: str = "residuos",
) -> dict:
    """Coordenar periodograma, janela, seleção, bootstrap e caracterização da série.

    Parameters
    ----------
    tempo : np.ndarray
        Tempos de observação em dias (JD), vetor de tamanho N.
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    periodo_minimo : float, default=0.1
        Menor período pesquisado, em dias; deve ser positivo.
    periodo_maximo : float, default=100.0
        Maior período solicitado, em dias; limitado ao intervalo observado.
    n_termos : int, default=4
        Ordem máxima de Fourier; ordem exata quando ordem_fixa=True.
    amostras_por_pico : int, default=10
        Sobreamostragem da grade em relação à escala 1/T.
    n_candidatos : int, default=15
        Número máximo de picos LS usados como sementes antes dos sub-harmônicos.
    n_bins : int, default=50
        Número de intervalos uniformes de fase, usados apenas na visualização.
    n_bootstrap : int, default=200
        Número de reamostragens solicitadas; zero desativa o bootstrap.
    semente : int, default=12345
        Semente passada a numpy.random.default_rng para reprodutibilidade.
    ordem_fixa : bool, default=False
        Se True, usa somente a ordem máxima; caso contrário, seleciona por BIC.
    metodo_bootstrap : str, default='residuos'
        residuos ou parametrico; outros valores causam ValueError.

    Returns
    -------
    dict
        Resultado completo: dados ordenados por tempo, referência e baseline;
        grades e potências; maior pico LS, FAP e níveis; melhor_ajuste;
        tabela_candidatos e tabela_janela; réplicas e resumo do bootstrap;
        dados ordenados por fase, médias por bin e modelo amostrado;
        amplitudes e parâmetros de Fourier. Consulte README.md para as chaves.

    Notes
    -----
    Aceita vetores conversíveis para float, finitos e de mesmo tamanho.
    Exige N > 2 + 2*n_termos, mesmo ao selecionar ordens menores. Ordena
    os tempos e limita o período máximo ao baseline, imprimindo um aviso
    quando o reduz. Não lê arquivos, não filtra banda e não aplica clipping.
    Lança ValueError para dados/parâmetros inválidos ou cobertura insuficiente;
    RuntimeError para periodograma não finito ou ausência de candidatos.
    Não salva nem exibe figuras. Equações e hipóteses: docs/METODOS.md.
    """
    tempo = np.asarray(tempo, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    erro = np.asarray(erro, dtype=float)
    if tempo.ndim != 1 or magnitude.ndim != 1 or erro.ndim != 1:
        raise ValueError("Tempo, magnitude e erro devem ser vetores unidimensionais.")
    if not (len(tempo) == len(magnitude) == len(erro)):
        raise ValueError("Tempo, magnitude e erro devem ter o mesmo tamanho.")
    if not (
        np.all(np.isfinite(tempo))
        and np.all(np.isfinite(magnitude))
        and np.all(np.isfinite(erro))
        and np.all(erro > 0)
    ):
        raise ValueError("Os dados devem ser finitos e todos os erros devem ser positivos.")
    if metodo_bootstrap not in {"residuos", "parametrico"}:
        raise ValueError("metodo_bootstrap deve ser 'residuos' ou 'parametrico'.")
    if periodo_minimo <= 0 or periodo_maximo <= periodo_minimo:
        raise ValueError("Use 0 < periodo_minimo < periodo_maximo.")
    if n_termos < 1 or amostras_por_pico < 5 or n_candidatos < 1 or n_bins < 5:
        raise ValueError("Parâmetros de ordem, grade, candidatos ou bins inválidos.")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap não pode ser negativo.")

    ordem_tempo = np.argsort(tempo)
    tempo = tempo[ordem_tempo]
    magnitude = magnitude[ordem_tempo]
    erro = erro[ordem_tempo]

    n_parametros = 1 + 2 * n_termos
    if len(tempo) <= n_parametros + 1:
        raise ValueError(
            f"São necessárias mais de {n_parametros + 1} observações "
            f"para um ajuste de Fourier de ordem {n_termos}."
        )

    baseline = float(np.ptp(tempo))
    if baseline <= 0:
        raise ValueError("As observações precisam cobrir mais de um instante.")

    periodo_maximo_efetivo = min(periodo_maximo, baseline)
    if periodo_maximo_efetivo < periodo_maximo:
        print(
            f"Período máximo reduzido para {periodo_maximo_efetivo:.6g} dias, "
            "igual ao intervalo observado."
        )
    if periodo_minimo >= periodo_maximo_efetivo:
        raise ValueError(
            "O intervalo observado é curto demais para os limites de período escolhidos."
        )

    tempo_referencia = float(np.min(tempo))
    tempo_relativo = tempo - tempo_referencia
    frequencia_minima = 1.0 / periodo_maximo_efetivo
    frequencia_maxima = 1.0 / periodo_minimo
    resolucao_frequencia = 1.0 / baseline

    ls, frequencias, potencia = calcular_periodograma(
        tempo_relativo,
        magnitude,
        erro,
        frequencia_minima,
        frequencia_maxima,
        amostras_por_pico,
    )

    frequencias_janela, potencia_janela, indices_janela, frequencias_alias = (
        analisar_janela_espectral(
            tempo_relativo,
            frequencias,
            frequencia_maxima,
            resolucao_frequencia,
        )
    )

    ajustes, origens = selecionar_candidatos(
        ls,
        frequencias,
        potencia,
        tempo_relativo,
        magnitude,
        erro,
        resolucao_frequencia,
        n_candidatos,
        n_termos,
        ordem_fixa,
        incluir_subharmonicos=True,
    )
    if not ajustes:
        raise RuntimeError("Não foi possível ajustar nenhum período candidato.")
    melhor_ajuste = ajustes[0]

    tabela_candidatos = criar_tabela_candidatos(
        ajustes,
        origens,
        ls,
        melhor_ajuste,
        frequencias_alias,
        resolucao_frequencia,
    )

    frequencia_pico_ls, potencia_pico_ls, fap_baluev, niveis_fap = calcular_significancia(
        ls,
        frequencias,
        potencia,
        amostras_por_pico,
        frequencia_minima,
        frequencia_maxima,
    )

    periodos_bootstrap = bootstrap_periodos(
        tempo_relativo,
        magnitude,
        erro,
        frequencias,
        resolucao_frequencia,
        n_termos,
        n_candidatos,
        n_bootstrap,
        semente,
        melhor_ajuste,
        ordem_fixa,
        metodo_bootstrap,
    )
    resumo_bootstrap = resumir_bootstrap(
        periodos_bootstrap,
        melhor_ajuste.periodo,
        resolucao_frequencia,
        n_bootstrap,
        metodo_bootstrap,
    )

    fase = np.remainder(tempo_relativo * melhor_ajuste.frequencia, 1.0)
    ordem_fase = np.argsort(fase)
    fase_ordenada = fase[ordem_fase]
    magnitude_ordenada = magnitude[ordem_fase]
    erro_ordenado = erro[ordem_fase]
    fase_media, magnitude_media, erro_medio = binning_ponderado(
        fase_ordenada,
        magnitude_ordenada,
        erro_ordenado,
        n_bins,
    )

    fase_modelo = np.linspace(0.0, 1.0, 1000)
    tempo_modelo = fase_modelo / melhor_ajuste.frequencia
    magnitude_modelo = (
        matriz_fourier(
            tempo_modelo,
            melhor_ajuste.frequencia,
            melhor_ajuste.n_termos,
        )
        @ melhor_ajuste.coeficientes
    )
    amplitude_pico_a_pico = float(np.ptp(magnitude_modelo))
    intervalo_percentil_90 = float(
        np.percentile(magnitude, 95) - np.percentile(magnitude, 5)
    )
    forma_fourier = parametros_fourier(melhor_ajuste.coeficientes)

    tabela_janela = criar_tabela_janela(
        frequencias_alias,
        potencia_janela,
        indices_janela,
    )

    return {
        "tempo": tempo,
        "tempo_relativo": tempo_relativo,
        "tempo_referencia": tempo_referencia,
        "magnitude": magnitude,
        "erro": erro,
        "baseline": baseline,
        "frequencias": frequencias,
        "potencia": potencia,
        "frequencias_janela": frequencias_janela,
        "potencia_janela": potencia_janela,
        "niveis_fap": niveis_fap,
        "fap_baluev": fap_baluev,
        "frequencia_pico_ls": frequencia_pico_ls,
        "periodo_pico_ls": 1.0 / frequencia_pico_ls,
        "potencia_pico_ls": potencia_pico_ls,
        "melhor_ajuste": melhor_ajuste,
        "tabela_candidatos": tabela_candidatos,
        "tabela_janela": tabela_janela,
        "resolucao_frequencia": resolucao_frequencia,
        "resumo_bootstrap": resumo_bootstrap,
        "periodos_bootstrap": periodos_bootstrap,
        "fase": fase_ordenada,
        "magnitude_fase": magnitude_ordenada,
        "erro_fase": erro_ordenado,
        "fase_media": fase_media,
        "magnitude_media": magnitude_media,
        "erro_medio": erro_medio,
        "fase_modelo": fase_modelo,
        "magnitude_modelo": magnitude_modelo,
        "amplitude_pico_a_pico": amplitude_pico_a_pico,
        "semamplitude_modelo": amplitude_pico_a_pico / 2.0,
        "intervalo_percentil_90": intervalo_percentil_90,
        "parametros_fourier": forma_fourier,
        "convencao_fase_fourier": (
            "A_k cos(2*pi*k*fase + phi_k); phi21 = (phi2 - 2*phi1) mod 2*pi"
        ),
    }
