"""Análise de curvas de luz de estrelas variáveis com Lomb-Scargle.

Principais etapas:
1. leitura e filtragem de observações AAVSO;
2. periodograma Lomb-Scargle ponderado e com média flutuante;
3. inspeção da função de janela e de possíveis aliases;
4. seleção de picos distintos e ajuste de Fourier ponderado aos dados individuais;
5. FAP de Baluev do maior pico e bootstrap condicional com janela fixa;
6. cálculo de parâmetros de forma e geração de gráficos diagnósticos.

Exemplo de uso:
    python codigo_estrelas.py observacoes.txt --banda V --observador SAH

Use ``python codigo_estrelas.py --help`` para ver todas as opções.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks


@dataclass
class AjusteFourier:
    frequencia: float
    n_termos: int
    coeficientes: np.ndarray
    modelo: np.ndarray
    residuos: np.ndarray
    rms_ponderado: float
    chi2: float
    chi2_reduzido: float
    bic: float
    posto: int
    numero_condicao: float

    @property
    def periodo(self) -> float:
        return 1.0 / self.frequencia


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estima e caracteriza o período de uma estrela variável."
    )
    parser.add_argument("arquivo", type=Path, help="Arquivo CSV ou TXT exportado da AAVSO.")
    parser.add_argument(
        "--banda",
        default="V",
        help="Banda fotométrica. Use TODOS para não filtrar.",
    )
    parser.add_argument(
        "--observador",
        default="SAH",
        help="Código do observador. Use TODOS para não filtrar.",
    )
    parser.add_argument(
        "--periodo-minimo",
        type=float,
        default=0.1,
        help="Menor período pesquisado, em dias (padrão: 0.1).",
    )
    parser.add_argument(
        "--periodo-maximo",
        type=float,
        default=100.0,
        help="Maior período pesquisado, em dias (padrão: 100).",
    )
    parser.add_argument(
        "--n-termos",
        type=int,
        default=4,
        help=(
            "Número máximo de harmônicos no ajuste de Fourier (padrão: 4). "
            "A ordem é escolhida por BIC, salvo se --ordem-fixa for usado."
        ),
    )
    parser.add_argument(
        "--ordem-fixa",
        action="store_true",
        help="Usa exatamente --n-termos, sem selecionar a ordem por BIC.",
    )
    parser.add_argument(
        "--amostras-por-pico",
        type=int,
        default=10,
        help="Sobreamostragem da grade de frequências (padrão: 10).",
    )
    parser.add_argument(
        "--n-candidatos",
        type=int,
        default=15,
        help="Número máximo de picos distintos examinados (padrão: 15).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Número de intervalos de fase usados apenas no gráfico (padrão: 50).",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=None,
        help=(
            "Corte robusto opcional em mediana e MAD. Por padrão não há corte, "
            "para não eliminar extremos físicos da curva de luz."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Número de reamostragens para avaliar o período (padrão: 200; 0 desativa).",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=12345,
        help="Semente aleatória do bootstrap (padrão: 12345).",
    )
    parser.add_argument(
        "--metodo-bootstrap",
        choices=("residuos", "parametrico"),
        default="residuos",
        help=(
            "Ruído das reamostragens: resíduos padronizados ou normal "
            "paramétrico (padrão: residuos)."
        ),
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Pasta opcional para salvar resumo, candidatos e gráficos.",
    )
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Não exibe nem salva gráficos.",
    )
    return parser


def validar_parametros(args: argparse.Namespace) -> None:
    if args.periodo_minimo <= 0 or args.periodo_maximo <= 0:
        raise ValueError("Os limites de período devem ser positivos.")
    if args.periodo_minimo >= args.periodo_maximo:
        raise ValueError("periodo-minimo deve ser menor que periodo-maximo.")
    if args.n_termos < 1:
        raise ValueError("n-termos deve ser pelo menos 1.")
    if args.amostras_por_pico < 5:
        raise ValueError("amostras-por-pico deve ser pelo menos 5.")
    if args.n_candidatos < 1:
        raise ValueError("n-candidatos deve ser pelo menos 1.")
    if args.n_bins < 5:
        raise ValueError("n-bins deve ser pelo menos 5.")
    if args.bootstrap < 0:
        raise ValueError("bootstrap não pode ser negativo.")
    if args.sigma_clip is not None and args.sigma_clip <= 0:
        raise ValueError("sigma-clip deve ser positivo.")


def carregar_dados(
    arquivo: Path,
    banda: str = "V",
    observador: str = "SAH",
    sigma_clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

    try:
        # ``sep=None`` reconhece, além do CSV usual da AAVSO, arquivos
        # delimitados por tabulação ou ponto e vírgula.
        dados = pd.read_csv(
            arquivo,
            comment="#",
            sep=None,
            engine="python",
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as erro_leitura:
        raise ValueError(
            f"Não foi possível interpretar o arquivo: {erro_leitura}"
        ) from erro_leitura
    dados.columns = dados.columns.str.strip()

    colunas_obrigatorias = {"JD", "Magnitude", "Uncertainty"}
    if banda.strip().upper() != "TODOS":
        colunas_obrigatorias.add("Band")
    if observador.strip().upper() != "TODOS":
        colunas_obrigatorias.add("Observer Code")

    ausentes = sorted(colunas_obrigatorias - set(dados.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(ausentes)}")

    n_lidas = len(dados)
    if banda.strip().upper() != "TODOS":
        banda_normalizada = banda.strip().upper()
        dados = dados[
            dados["Band"].astype(str).str.strip().str.upper() == banda_normalizada
        ]
    if observador.strip().upper() != "TODOS":
        observador_normalizado = observador.strip().upper()
        dados = dados[
            dados["Observer Code"].astype(str).str.strip().str.upper()
            == observador_normalizado
        ]
    n_filtradas = len(dados)

    dados = dados.copy()
    for coluna in ("JD", "Magnitude", "Uncertainty"):
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    valores = dados[["JD", "Magnitude", "Uncertainty"]].to_numpy(dtype=float)
    validos = np.all(np.isfinite(valores), axis=1) & (valores[:, 2] > 0)
    n_invalidos = int(np.count_nonzero(~validos))
    valores = valores[validos]

    if len(valores) == 0:
        raise ValueError(
            "Nenhuma observação possui JD, magnitude e incerteza finitos, "
            "com incerteza estritamente positiva."
        )

    tempo, magnitude, erro = valores.T
    ordem = np.argsort(tempo)
    tempo, magnitude, erro = tempo[ordem], magnitude[ordem], erro[ordem]

    n_clip = 0
    if sigma_clip is not None:
        mediana = np.median(magnitude)
        mad = 1.4826 * np.median(np.abs(magnitude - mediana))
        if mad > 0:
            manter = np.abs(magnitude - mediana) <= sigma_clip * mad
            n_clip = int(np.count_nonzero(~manter))
            tempo, magnitude, erro = tempo[manter], magnitude[manter], erro[manter]

    print(f"Observações lidas: {n_lidas}")
    print(f"Após os filtros de banda/observador: {n_filtradas}")
    print(f"Removidas por dados ou incertezas inválidos: {n_invalidos}")
    if sigma_clip is None:
        print("Corte por magnitude: desativado")
    else:
        print(f"Removidas pelo corte robusto de magnitude: {n_clip}")
    print(f"Observações usadas: {len(tempo)}")
    n_instantes = int(np.unique(tempo).size)
    if n_instantes < len(tempo):
        print(
            f"Aviso: há {len(tempo) - n_instantes} observações com JD repetido; "
            "elas são preservadas como medidas independentes."
        )

    return tempo, magnitude, erro


def matriz_fourier(
    tempo_relativo: np.ndarray,
    frequencia: float,
    n_termos: int,
) -> np.ndarray:
    fase = np.remainder(tempo_relativo * frequencia, 1.0)
    colunas = [np.ones_like(fase)]
    for k in range(1, n_termos + 1):
        angulo = 2.0 * np.pi * k * fase
        colunas.extend((np.sin(angulo), np.cos(angulo)))
    return np.column_stack(colunas)


def avaliar_fourier(
    tempo_relativo: np.ndarray,
    magnitude: np.ndarray,
    erro: np.ndarray,
    frequencia: float,
    n_termos: int,
) -> AjusteFourier:
    if not np.isfinite(frequencia) or frequencia <= 0:
        raise ValueError("A frequência do ajuste deve ser finita e positiva.")
    matriz = matriz_fourier(tempo_relativo, frequencia, n_termos)
    matriz_ponderada = matriz / erro[:, None]
    alvo_ponderado = magnitude / erro
    coeficientes, _, posto, valores_singulares = np.linalg.lstsq(
        matriz_ponderada,
        alvo_ponderado,
        rcond=None,
    )
    n_parametros = matriz.shape[1]
    if posto < n_parametros:
        raise np.linalg.LinAlgError(
            f"Matriz de Fourier sem posto completo ({posto}/{n_parametros})."
        )
    if valores_singulares[-1] <= 0:
        raise np.linalg.LinAlgError("Matriz de Fourier numericamente singular.")
    numero_condicao = float(valores_singulares[0] / valores_singulares[-1])
    modelo = matriz @ coeficientes
    residuos = magnitude - modelo
    pesos = 1.0 / erro**2
    chi2 = float(np.sum((residuos / erro) ** 2))
    graus_liberdade = len(magnitude) - len(coeficientes)
    chi2_reduzido = chi2 / graus_liberdade if graus_liberdade > 0 else np.nan
    rms_ponderado = float(np.sqrt(np.sum(pesos * residuos**2) / np.sum(pesos)))
    bic = chi2 + n_parametros * np.log(len(magnitude))
    if not np.all(np.isfinite(coeficientes)) or not np.isfinite(bic):
        raise np.linalg.LinAlgError("O ajuste de Fourier produziu valores não finitos.")
    return AjusteFourier(
        frequencia=float(frequencia),
        n_termos=n_termos,
        coeficientes=coeficientes,
        modelo=modelo,
        residuos=residuos,
        rms_ponderado=rms_ponderado,
        chi2=chi2,
        chi2_reduzido=float(chi2_reduzido),
        bic=float(bic),
        posto=int(posto),
        numero_condicao=numero_condicao,
    )


def indices_picos_distintos(
    frequencias: np.ndarray,
    potencia: np.ndarray,
    resolucao_frequencia: float,
    quantidade: int,
) -> np.ndarray:
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
    meia_largura = 0.5 * resolucao_frequencia
    limite_inferior = max(frequencia_minima, frequencia_inicial - meia_largura)
    limite_superior = min(frequencia_maxima, frequencia_inicial + meia_largura)

    ordens = [n_termos_maximo] if ordem_fixa else range(1, n_termos_maximo + 1)
    ajustes: list[AjusteFourier] = []

    for n_termos in ordens:
        def objetivo(frequencia: float) -> float:
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


def calcular_janela_espectral(
    tempo_relativo: np.ndarray,
    frequencias: np.ndarray,
) -> np.ndarray:
    janela = LombScargle(
        tempo_relativo,
        np.ones_like(tempo_relativo),
        fit_mean=False,
        center_data=False,
        normalization="standard",
    )
    return janela.power(frequencias, method="auto")


def classificar_relacao(
    frequencia: float,
    frequencia_principal: float,
    frequencias_janela: np.ndarray,
    resolucao_frequencia: float,
) -> str:
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


def resumir_bootstrap(
    periodos: np.ndarray,
    periodo_principal: float,
    resolucao_frequencia: float,
    n_solicitados: int,
    metodo: str,
) -> dict:
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
            ls_b = LombScargle(
                tempo_relativo,
                magnitude_b,
                dy=erro,
                fit_mean=True,
                center_data=True,
                nterms=1,
                normalization="standard",
            )
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


def parametros_fourier(coeficientes: np.ndarray) -> dict[str, float]:
    if len(coeficientes) < 3:
        return {"A1": np.nan, "A2": np.nan, "R21": np.nan, "phi21": np.nan}

    a1, b1 = coeficientes[1], coeficientes[2]
    amplitude_1 = float(np.hypot(a1, b1))
    if len(coeficientes) < 5:
        return {"A1": amplitude_1, "A2": np.nan, "R21": np.nan, "phi21": np.nan}

    a2, b2 = coeficientes[3], coeficientes[4]
    amplitude_2 = float(np.hypot(a2, b2))
    escala = max(1.0, float(np.max(np.abs(coeficientes))))
    tolerancia = 100.0 * np.finfo(float).eps * escala
    if amplitude_1 <= tolerancia or amplitude_2 <= tolerancia:
        return {
            "A1": amplitude_1,
            "A2": amplitude_2,
            "R21": np.nan if amplitude_1 <= tolerancia else amplitude_2 / amplitude_1,
            "phi21": np.nan,
        }
    # Convenção astronômica de série em cossenos:
    # A_k cos(2πkφ + φ_k). Como o modelo interno é a_k sin + b_k cos,
    # φ_k = atan2(-a_k, b_k).
    fase_1 = float(np.arctan2(-a1, b1))
    fase_2 = float(np.arctan2(-a2, b2))
    r21 = amplitude_2 / amplitude_1
    phi21 = float(np.mod(fase_2 - 2.0 * fase_1, 2.0 * np.pi))
    return {"A1": amplitude_1, "A2": amplitude_2, "R21": r21, "phi21": phi21}


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

    ls = LombScargle(
        tempo_relativo,
        magnitude,
        dy=erro,
        fit_mean=True,
        center_data=True,
        nterms=1,
        normalization="standard",
    )
    frequencias = ls.autofrequency(
        samples_per_peak=amostras_por_pico,
        minimum_frequency=frequencia_minima,
        maximum_frequency=frequencia_maxima,
    )
    potencia = ls.power(frequencias, method="auto")

    if not np.all(np.isfinite(potencia)):
        raise RuntimeError("O periodograma produziu potências não finitas.")

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

    linhas_candidatos = []
    bic_minimo = min(ajuste.bic for ajuste in ajustes)
    for posicao, (ajuste, origem) in enumerate(zip(ajustes, origens), start=1):
        linhas_candidatos.append(
            {
                "ordem": posicao,
                "periodo_dias": ajuste.periodo,
                "frequencia_dia-1": ajuste.frequencia,
                "potencia_LS": float(ls.power(ajuste.frequencia)),
                "RMS_ponderado": ajuste.rms_ponderado,
                "chi2_reduzido": ajuste.chi2_reduzido,
                "BIC": ajuste.bic,
                "delta_BIC": ajuste.bic - bic_minimo,
                "n_harmonicos": ajuste.n_termos,
                "numero_condicao": ajuste.numero_condicao,
                "origem": origem,
                "selecionado": posicao == 1,
                "relacao": classificar_relacao(
                    ajuste.frequencia,
                    melhor_ajuste.frequencia,
                    frequencias_alias,
                    resolucao_frequencia,
                ),
            }
        )
    tabela_candidatos = pd.DataFrame(linhas_candidatos)

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

    tabela_janela = pd.DataFrame(
        {
            "frequencia_dia-1": frequencias_alias,
            "periodo_dias": 1.0 / frequencias_alias,
            "potencia_janela": potencia_janela[indices_janela],
        }
    ).sort_values("potencia_janela", ascending=False, ignore_index=True)

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


def imprimir_resultados(resultado: dict) -> None:
    ajuste: AjusteFourier = resultado["melhor_ajuste"]
    print("\n==============================")
    print("RESULTADO PRINCIPAL")
    print("==============================")
    print(f"Intervalo observado: {resultado['baseline']:.6f} dias")
    print(f"Pontos na grade de frequências: {len(resultado['frequencias'])}")
    print(f"Período selecionado: {ajuste.periodo:.10f} dias")
    print(f"Frequência selecionada: {ajuste.frequencia:.10f} dia^-1")
    print(f"Ordem de Fourier selecionada: {ajuste.n_termos} harmônico(s)")
    print(f"RMS ponderado nos dados individuais: {ajuste.rms_ponderado:.6f} mag")
    print(f"Qui-quadrado reduzido: {ajuste.chi2_reduzido:.6f}")
    print(
        "Maior pico Lomb-Scargle: "
        f"P = {resultado['periodo_pico_ls']:.10f} dias, "
        f"potência = {resultado['potencia_pico_ls']:.6g}"
    )
    print(
        "FAP de Baluev do maior pico Lomb-Scargle: "
        f"{resultado['fap_baluev']:.6g}"
    )
    print(
        "Nota: a FAP quantifica picos produzidos por ruído sem sinal periódico; "
        "não é a probabilidade de a estrela ser aperiódica."
    )
    print(f"Amplitude pico a pico do modelo: {resultado['amplitude_pico_a_pico']:.6f} mag")
    print(f"Semiamplitude do modelo: {resultado['semamplitude_modelo']:.6f} mag")
    print(
        "Intervalo P95-P5 das observações: "
        f"{resultado['intervalo_percentil_90']:.6f} mag"
    )
    for nome, valor in resultado["parametros_fourier"].items():
        unidade = " (rad)" if nome == "phi21" else ""
        print(f"{nome}{unidade}: {valor:.6f}")
    print(f"Convenção de fase: {resultado['convencao_fase_fourier']}")

    print(
        "\nCandidatos distintos (Fourier selecionado por BIC; ΔBIC < 2 "
        "é desempatatado pela potência Lomb-Scargle):"
    )
    colunas = [
        "ordem",
        "periodo_dias",
        "frequencia_dia-1",
        "potencia_LS",
        "RMS_ponderado",
        "chi2_reduzido",
        "delta_BIC",
        "n_harmonicos",
        "origem",
        "relacao",
    ]
    print(
        resultado["tabela_candidatos"][colunas].to_string(
            index=False,
            float_format=lambda valor: f"{valor:.7g}",
        )
    )

    print("\nPicos dominantes da função de janela:")
    print(
        resultado["tabela_janela"].to_string(
            index=False,
            float_format=lambda valor: f"{valor:.7g}",
        )
    )

    bootstrap = resultado["resumo_bootstrap"]
    if bootstrap["n_validos"]:
        print("\nBootstrap condicional do período (tempos fixos):")
        print(f"Método de ruído: {bootstrap['metodo']}")
        print(
            f"Reamostragens válidas: {bootstrap['n_validos']}/"
            f"{bootstrap['n_solicitados']}"
        )
        if bootstrap["fracao_valida"] < 0.95:
            print(
                "Aviso: muitas reamostragens falharam; o resumo do bootstrap "
                "pode não ser representativo."
            )
        print(
            "Fração que recuperou o mesmo pico: "
            f"{bootstrap['fracao_mesmo_pico']:.1%}"
        )
        intervalo = bootstrap["intervalo_condicional"]
        if intervalo is not None:
            print(
                "Intervalo condicional P16-mediana-P84: "
                f"{intervalo['p16']:.10f} - {intervalo['mediana']:.10f} - "
                f"{intervalo['p84']:.10f} dias"
            )
        print("Modos mais frequentes do bootstrap:")
        for modo in bootstrap["modos"]:
            print(
                f"  P = {modo['periodo_mediano']:.10f} dias | "
                f"fração = {modo['fracao']:.1%}"
            )
        if bootstrap["fracao_mesmo_pico"] < 0.8:
            print(
                "Aviso: a distribuição é multimodal; um único erro simétrico "
                "para o período seria enganoso."
            )


def criar_graficos(resultado: dict) -> dict[str, plt.Figure]:
    figuras: dict[str, plt.Figure] = {}

    figura_curva, eixo = plt.subplots(figsize=(9, 5))
    eixo.errorbar(
        resultado["tempo"],
        resultado["magnitude"],
        yerr=resultado["erro"],
        fmt=".",
        markersize=3,
        alpha=0.55,
        elinewidth=0.5,
        capsize=0,
    )
    eixo.invert_yaxis()
    eixo.set(xlabel="Tempo (JD)", ylabel="Magnitude", title="Curva de luz")
    figura_curva.tight_layout()
    figuras["curva_luz"] = figura_curva

    figura_periodograma, (eixo_sinal, eixo_janela) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )
    periodos = 1.0 / resultado["frequencias"]
    ordem = np.argsort(periodos)
    eixo_sinal.plot(periodos[ordem], resultado["potencia"][ordem], linewidth=0.8)
    cores = {0.10: "#d99b00", 0.05: "#d65f00", 0.01: "#b00020"}
    for probabilidade, nivel in resultado["niveis_fap"].items():
        eixo_sinal.axhline(
            nivel,
            color=cores.get(probabilidade, "gray"),
            linestyle="--",
            linewidth=1,
            label=f"FAP = {probabilidade:.0%}",
        )
    periodo_principal = resultado["melhor_ajuste"].periodo
    eixo_sinal.axvline(
        resultado["periodo_pico_ls"],
        color="tab:gray",
        linestyle=":",
        linewidth=1.3,
        label="Maior pico LS",
    )
    eixo_sinal.axvline(
        periodo_principal,
        color="black",
        linewidth=1.2,
        label="Modelo selecionado",
    )
    eixo_sinal.set(
        xlabel="Período (dias)",
        ylabel="Potência Lomb-Scargle",
        title="Periodograma e níveis de significância",
    )
    eixo_sinal.set_xscale("log")
    eixo_sinal.set_xlim(periodos.min(), periodos.max())
    eixo_sinal.legend(loc="best")

    periodos_janela = 1.0 / resultado["frequencias_janela"]
    ordem_janela = np.argsort(periodos_janela)
    eixo_janela.plot(
        periodos_janela[ordem_janela],
        resultado["potencia_janela"][ordem_janela],
        color="tab:purple",
        linewidth=0.8,
    )
    eixo_janela.set(
        xlabel="Período (dias)",
        ylabel="Potência da janela",
        title="Função de janela observacional",
    )
    eixo_janela.set_xscale("log")
    eixo_janela.set_xlim(periodos.min(), min(resultado["baseline"], periodos_janela.max()))
    figura_periodograma.tight_layout()
    figuras["periodograma_janela"] = figura_periodograma

    figura_fase, eixo = plt.subplots(figsize=(11, 7))
    eixo.errorbar(
        resultado["fase"],
        resultado["magnitude_fase"],
        yerr=resultado["erro_fase"],
        fmt=".",
        markersize=4,
        alpha=0.22,
        color="tab:blue",
        elinewidth=0.4,
        label="Observações",
    )
    if len(resultado["fase_media"]):
        eixo.errorbar(
            resultado["fase_media"],
            resultado["magnitude_media"],
            yerr=resultado["erro_medio"],
            fmt="o",
            markersize=4,
            color="tab:red",
            capsize=2,
            label="Médias ponderadas por bin",
        )
    eixo.plot(
        resultado["fase_modelo"],
        resultado["magnitude_modelo"],
        color="tab:green",
        linewidth=2.5,
        label="Fourier ponderado nos dados individuais",
    )
    eixo.invert_yaxis()
    eixo.set(
        xlabel="Fase",
        ylabel="Magnitude",
        title=f"Curva dobrada - P = {periodo_principal:.10f} dias",
    )
    eixo.legend(loc="best")
    figura_fase.tight_layout()
    figuras["curva_fase"] = figura_fase

    return figuras


def valor_json(valor):
    if isinstance(valor, dict):
        return {chave: valor_json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [valor_json(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return valor_json(valor.tolist())
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        valor_float = float(valor)
        return valor_float if np.isfinite(valor_float) else None
    return valor


def salvar_resultados(
    resultado: dict,
    pasta: Path,
    figuras: dict[str, plt.Figure],
) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    resultado["tabela_candidatos"].to_csv(pasta / "periodos_candidatos.csv", index=False)
    resultado["tabela_janela"].to_csv(pasta / "picos_janela.csv", index=False)
    pd.DataFrame(
        {"periodo_dias": resultado["periodos_bootstrap"]}
    ).to_csv(pasta / "periodos_bootstrap.csv", index=False)

    ajuste: AjusteFourier = resultado["melhor_ajuste"]
    resumo = {
        "n_observacoes": len(resultado["tempo"]),
        "intervalo_observado_dias": resultado["baseline"],
        "resolucao_frequencia_dia-1": resultado["resolucao_frequencia"],
        "periodo_dias": ajuste.periodo,
        "frequencia_dia-1": ajuste.frequencia,
        "n_harmonicos_fourier": ajuste.n_termos,
        "BIC": ajuste.bic,
        "rms_ponderado_magnitude": ajuste.rms_ponderado,
        "chi2_reduzido": ajuste.chi2_reduzido,
        "maior_pico_lomb_scargle": {
            "periodo_dias": resultado["periodo_pico_ls"],
            "frequencia_dia-1": resultado["frequencia_pico_ls"],
            "potencia": resultado["potencia_pico_ls"],
            "fap_baluev": resultado["fap_baluev"],
        },
        "amplitude_pico_a_pico_magnitude": resultado["amplitude_pico_a_pico"],
        "semamplitude_magnitude": resultado["semamplitude_modelo"],
        "intervalo_percentil_90_magnitude": resultado["intervalo_percentil_90"],
        "parametros_fourier": resultado["parametros_fourier"],
        "convencao_fase_fourier": resultado["convencao_fase_fourier"],
        "bootstrap": resultado["resumo_bootstrap"],
    }
    with (pasta / "resumo.json").open("w", encoding="utf-8") as arquivo:
        json.dump(valor_json(resumo), arquivo, ensure_ascii=False, indent=2)

    for nome, figura in figuras.items():
        figura.savefig(pasta / f"{nome}.png", dpi=180, bbox_inches="tight")


def main() -> None:
    parser = criar_parser()
    args = parser.parse_args()
    try:
        validar_parametros(args)
        tempo, magnitude, erro = carregar_dados(
            args.arquivo,
            banda=args.banda,
            observador=args.observador,
            sigma_clip=args.sigma_clip,
        )
        resultado = analisar_serie(
            tempo,
            magnitude,
            erro,
            periodo_minimo=args.periodo_minimo,
            periodo_maximo=args.periodo_maximo,
            n_termos=args.n_termos,
            amostras_por_pico=args.amostras_por_pico,
            n_candidatos=args.n_candidatos,
            n_bins=args.n_bins,
            n_bootstrap=args.bootstrap,
            semente=args.semente,
            ordem_fixa=args.ordem_fixa,
            metodo_bootstrap=args.metodo_bootstrap,
        )
        imprimir_resultados(resultado)

        figuras: dict[str, plt.Figure] = {}
        if not args.sem_graficos:
            figuras = criar_graficos(resultado)
        if args.saida is not None:
            salvar_resultados(resultado, args.saida, figuras)
            print(f"\nResultados salvos em: {args.saida.resolve()}")
        if figuras and args.saida is None:
            plt.show()
        elif figuras:
            plt.close("all")
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as erro_execucao:
        parser.exit(2, f"Erro: {erro_execucao}\n")


if __name__ == "__main__":
    main()
