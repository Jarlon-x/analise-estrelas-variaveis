"""Significância de Baluev, bootstrap condicional e binning ponderado."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from fourier import AjusteFourier
from lomb_scargle import criar_lomb_scargle
from selecao_modelos import selecionar_candidatos


def calcular_significancia(
    ls: LombScargle,
    frequencias: np.ndarray,
    potencia: np.ndarray,
    amostras_por_pico: int,
    frequencia_minima: float,
    frequencia_maxima: float,
) -> tuple[float, float, float, dict[float, float]]:
    """Estimar a FAP de Baluev do maior pico da grade Lomb–Scargle.

    Parameters
    ----------
    ls : LombScargle
        Objeto LombScargle ponderado construído para os dados analisados.
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.
    potencia : np.ndarray
        Potência Lomb–Scargle adimensional, alinhada à grade de frequências.
    amostras_por_pico : int
        Sobreamostragem da grade em relação à escala 1/T.
    frequencia_minima : float
        Limite inferior da busca em dia^-1.
    frequencia_maxima : float
        Limite superior da busca em dia^-1.

    Returns
    -------
    tuple
        (frequencia_pico_ls, potencia_pico_ls, fap_baluev, niveis_fap).
        niveis_fap associa probabilidades 0.10, 0.05 e 0.01 às suas potências.

    Notes
    -----
    A FAP avalia o maior pico sinusoidal sob a hipótese nula, não o modelo
    Fourier escolhido por BIC. Se Astropy lançar ValueError, FloatingPointError
    ou OverflowError, retorna FAP numpy.nan e níveis {}, preservando o pico.
    """
    indice_pico_ls = int(np.argmax(potencia))
    frequencia_pico_ls = float(frequencias[indice_pico_ls])
    potencia_pico_ls = float(potencia[indice_pico_ls])
    try:
        fap_baluev = float(
            ls.false_alarm_probability(
                potencia_pico_ls,
                method="baluev",
                samples_per_peak=amostras_por_pico,
                minimum_frequency=frequencia_minima,
                maximum_frequency=frequencia_maxima,
            )
        )
        probabilidades_fap = np.array([0.10, 0.05, 0.01])
        niveis_fap_valores = ls.false_alarm_level(
            probabilidades_fap,
            method="baluev",
            samples_per_peak=amostras_por_pico,
            minimum_frequency=frequencia_minima,
            maximum_frequency=frequencia_maxima,
        )
        niveis_fap = {
            float(probabilidade): float(nivel)
            for probabilidade, nivel in zip(probabilidades_fap, niveis_fap_valores)
        }
    except (ValueError, FloatingPointError, OverflowError):
        fap_baluev = np.nan
        niveis_fap = {}
    return frequencia_pico_ls, potencia_pico_ls, fap_baluev, niveis_fap


def resumir_bootstrap(
    periodos: np.ndarray,
    periodo_principal: float,
    resolucao_frequencia: float,
    n_solicitados: int,
    metodo: str,
) -> dict:
    """Resumir a validade, recuperação do pico e modos dos períodos reamostrados.

    Parameters
    ----------
    periodos : np.ndarray
        Períodos válidos do bootstrap, em dias; pode ser um vetor vazio.
    periodo_principal : float
        Período do modelo selecionado nos dados originais, em dias.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.
    n_solicitados : int
        Número total de reamostragens solicitadas, incluindo falhas.
    metodo : str
        Nome do método registrado no resumo, normalmente residuos ou parametrico.

    Returns
    -------
    dict
        Método, janela_temporal_fixa, contagens, frações, intervalo_condicional
        (p16, mediana, p84) e até cinco modos com período, fração e contagem.

    Notes
    -----
    Define mesmo pico por abs(1/periodo - 1/periodo_principal) <= resolucao.
    Calcula percentis somente com pelo menos cinco recuperações do pico.
    Agrupa frequências ordenadas pela distância à mediana do último grupo.
    Com vetor vazio, usa NaN para frações indefinidas e None para intervalo.
    Para vetor não vazio, n_solicitados deve ser positivo.
    """
    if len(periodos) == 0:
        return {
            "metodo": metodo,
            "janela_temporal_fixa": True,
            "n_solicitados": int(n_solicitados),
            "n_validos": 0,
            "fracao_valida": 0.0 if n_solicitados else np.nan,
            "fracao_mesmo_pico": np.nan,
            "intervalo_condicional": None,
            "modos": [],
        }

    frequencias = 1.0 / periodos
    frequencia_principal = 1.0 / periodo_principal
    mesmo_pico = np.abs(frequencias - frequencia_principal) <= resolucao_frequencia
    fracao = float(np.mean(mesmo_pico))

    intervalo = None
    if np.count_nonzero(mesmo_pico) >= 5:
        q16, q50, q84 = np.percentile(periodos[mesmo_pico], [16, 50, 84])
        intervalo = {
            "p16": float(q16),
            "mediana": float(q50),
            "p84": float(q84),
        }

    ordem = np.argsort(frequencias)
    frequencias_ordenadas = frequencias[ordem]
    grupos: list[list[float]] = []
    for frequencia in frequencias_ordenadas:
        if not grupos or frequencia - np.median(grupos[-1]) > resolucao_frequencia:
            grupos.append([float(frequencia)])
        else:
            grupos[-1].append(float(frequencia))

    modos = sorted(grupos, key=len, reverse=True)[:5]
    resumo_modos = [
        {
            "periodo_mediano": float(1.0 / np.median(grupo)),
            "fracao": len(grupo) / len(frequencias),
            "contagem": len(grupo),
        }
        for grupo in modos
    ]
    return {
        "metodo": metodo,
        "janela_temporal_fixa": True,
        "n_solicitados": int(n_solicitados),
        "n_validos": int(len(periodos)),
        "fracao_valida": float(len(periodos) / n_solicitados),
        "fracao_mesmo_pico": fracao,
        "intervalo_condicional": intervalo,
        "modos": resumo_modos,
    }


def bootstrap_periodos(
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    frequencias: np.ndarray,
    resolucao_frequencia: float,
    n_termos: int,
    n_candidatos: int,
    n_bootstrap: int,
    semente: int,
    ajuste_base: AjusteFourier,
    ordem_fixa: bool,
    metodo: str,
) -> np.ndarray:
    """Reamostrar magnitudes em torno do modelo mantendo tempos e erros fixos.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    frequencias : np.ndarray
        Grade crescente e regularmente espaçada de frequências em dia^-1.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.
    n_termos : int
        Ordem máxima reavaliada em cada réplica, ou ordem exata se ordem_fixa.
    n_candidatos : int
        Número máximo de picos LS usados como sementes antes dos sub-harmônicos.
    n_bootstrap : int
        Número de reamostragens solicitadas; zero desativa o bootstrap.
    semente : int
        Semente passada a numpy.random.default_rng para reprodutibilidade.
    ajuste_base : AjusteFourier
        Modelo selecionado nos dados originais, com modelo e resíduos alinhados.
    ordem_fixa : bool
        Se True, usa somente a ordem máxima; caso contrário, seleciona por BIC.
    metodo : str
        residuos para reamostragem empírica; parametrico para ruído normal padrão.

    Returns
    -------
    numpy.ndarray
        Períodos selecionados, em dias, apenas das reamostragens válidas.
        O tamanho pode ser menor que n_bootstrap; zero produz um vetor vazio.

    Notes
    -----
    No método residuos, centraliza resíduos/erro e aplica sqrt(N/(N-p))
    quando N-p > 0; sorteia com reposição. No método parametrico, sorteia
    normais padrão. Reexecuta a seleção com até min(n_candidatos, 8) sementes
    LS e sub-harmônicos. Falhas numéricas tratadas são omitidas.
    A validação do nome do método é feita por analisar_serie; nesta função,
    qualquer nome diferente de parametrico segue o ramo de resíduos.
    """
    if n_bootstrap == 0:
        return np.array([], dtype=float)

    gerador = np.random.default_rng(semente)
    periodos: list[float] = []
    residuos_padronizados = ajuste_base.residuos / erro
    residuos_padronizados = residuos_padronizados - np.mean(residuos_padronizados)
    graus_liberdade = len(magnitude) - len(ajuste_base.coeficientes)
    if graus_liberdade > 0:
        residuos_padronizados *= np.sqrt(len(magnitude) / graus_liberdade)

    for _ in range(n_bootstrap):
        if metodo == "parametrico":
            ruido_padronizado = gerador.normal(size=len(magnitude))
        else:
            ruido_padronizado = gerador.choice(
                residuos_padronizados,
                size=len(magnitude),
                replace=True,
            )
        # Os tempos e as incertezas ficam fixos: cada repetição preserva a
        # função de janela observacional e testa a troca entre aliases.
        magnitude_b = ajuste_base.modelo + erro * ruido_padronizado
        try:
            ls_b = criar_lomb_scargle(tempo_relativo, magnitude_b, erro)
            potencia_b = ls_b.power(frequencias, method="auto")
            ajustes_b, _ = selecionar_candidatos(
                ls_b,
                frequencias,
                potencia_b,
                tempo_relativo,
                magnitude_b,
                erro,
                resolucao_frequencia,
                min(n_candidatos, 8),
                n_termos,
                ordem_fixa,
                incluir_subharmonicos=True,
            )
            if ajustes_b:
                periodos.append(ajustes_b[0].periodo)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

    return np.asarray(periodos, dtype=float)


def binning_ponderado(
    fase: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agrupar observações em fase para visualização com pesos 1/erro**2.

    Parameters
    ----------
    fase : np.ndarray
        Fases das observações no intervalo [0, 1), vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    n_bins : int
        Número de intervalos uniformes de fase, usados apenas na visualização.

    Returns
    -------
    tuple of numpy.ndarray
        (fases_medias, magnitudes_medias, erros_medios); inclui somente bins
        com pelo menos duas observações e preserva a ordem dos bins.

    Notes
    -----
    O erro de cada média é o máximo entre o erro formal e a dispersão
    ponderada dividida pela raiz do tamanho efetivo da amostra.
    Esses dados agrupados não são usados no ajuste de Fourier.
    """
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(fase, limites) - 1, 0, n_bins - 1)
    fases_medias: list[float] = []
    magnitudes_medias: list[float] = []
    erros_medios: list[float] = []

    for indice_bin in range(n_bins):
        mascara = indices == indice_bin
        # Um bin com apenas uma observação não é uma média e apenas duplicaria
        # visualmente o dado individual.
        if np.count_nonzero(mascara) < 2:
            continue
        pesos = 1.0 / erro[mascara] ** 2
        fase_media = float(np.average(fase[mascara], weights=pesos))
        magnitude_media = float(np.average(magnitude[mascara], weights=pesos))
        erro_formal = float(np.sqrt(1.0 / np.sum(pesos)))
        n_efetivo = float(np.sum(pesos) ** 2 / np.sum(pesos**2))
        dispersao = float(
            np.sqrt(
                np.sum(pesos * (magnitude[mascara] - magnitude_media) ** 2)
                / np.sum(pesos)
            )
        )
        erro_dispersao = dispersao / np.sqrt(max(n_efetivo, 1.0))
        fases_medias.append(fase_media)
        magnitudes_medias.append(magnitude_media)
        erros_medios.append(max(erro_formal, erro_dispersao))

    return (
        np.asarray(fases_medias),
        np.asarray(magnitudes_medias),
        np.asarray(erros_medios),
    )
