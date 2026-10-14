"""Tabelas de resultados, resumo no terminal e exportação CSV, JSON e PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from fourier import AjusteFourier
from janela_espectral import classificar_relacao


def criar_tabela_candidatos(
    ajustes: list[AjusteFourier],
    origens: list[str],
    ls: LombScargle,
    melhor_ajuste: AjusteFourier,
    frequencias_alias: np.ndarray,
    resolucao_frequencia: float,
) -> pd.DataFrame:
    """Montar a tabela dos modelos na ordem definida pela seleção.

    Parameters
    ----------
    ajustes : list[AjusteFourier]
        Lista não vazia retornada por selecionar_candidatos.
    origens : list[str]
        Rótulos das sementes, na mesma ordem e quantidade que ajustes.
    ls : LombScargle
        Objeto LombScargle ponderado construído para os dados analisados.
    melhor_ajuste : AjusteFourier
        Modelo selecionado, esperado como ajustes[0].
    frequencias_alias : np.ndarray
        Frequências dos picos selecionados da janela, em dia^-1.
    resolucao_frequencia : float
        Escala 1/T em dia^-1, sendo T o intervalo temporal observado.

    Returns
    -------
    pandas.DataFrame
        Uma linha por candidato, com período, frequência, potência LS, RMS,
        chi2 reduzido, BIC, delta_BIC, ordem de Fourier, condição e rótulos.

    Notes
    -----
    A primeira linha é marcada como selecionada. delta_BIC usa o mínimo
    global da lista, que pode diferir do BIC do candidato selecionado.
    """
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
    return tabela_candidatos


def criar_tabela_janela(
    frequencias_alias: np.ndarray,
    potencia_janela: np.ndarray,
    indices_janela: np.ndarray,
) -> pd.DataFrame:
    """Organizar os picos da janela em uma tabela por potência decrescente.

    Parameters
    ----------
    frequencias_alias : np.ndarray
        Frequências dos picos selecionados da janela, em dia^-1.
    potencia_janela : np.ndarray
        Potência adimensional da janela em sua grade completa.
    indices_janela : np.ndarray
        Índices dos picos selecionados na grade da janela.

    Returns
    -------
    pandas.DataFrame
        Colunas frequencia_dia-1, periodo_dias e potencia_janela, com índice reiniciado.

    Notes
    -----
    frequencias_alias deve corresponder a indices_janela na grade original.
    """
    tabela_janela = pd.DataFrame(
        {
            "frequencia_dia-1": frequencias_alias,
            "periodo_dias": 1.0 / frequencias_alias,
            "potencia_janela": potencia_janela[indices_janela],
        }
    ).sort_values("potencia_janela", ascending=False, ignore_index=True)
    return tabela_janela


def imprimir_resultados(resultado: dict) -> None:
    """Exibir resumo, candidatos, janela e diagnóstico de bootstrap no terminal.

    Parameters
    ----------
    resultado : dict
        Dicionário completo retornado por analise.analisar_serie.

    Returns
    -------
    None
        O resultado é escrito na saída padrão.

    Notes
    -----
    Preserva os formatos numéricos e mensagens da versão de referência.
    O bloco de bootstrap só é impresso quando existe alguma réplica válida.
    """
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


def valor_json(valor):
    """Converter recursivamente os tipos usados no resumo para serialização JSON.

    Parameters
    ----------
    valor : object
        Escalar, array, lista, tupla ou dicionário presente no resumo.

    Returns
    -------
    object
        Valor convertido: arrays/tuplas viram listas, inteiros NumPy viram int,
        reais viram float e reais não finitos viram None (null no JSON).

    Notes
    -----
    Dicionários têm seus valores convertidos; tipos não reconhecidos são
    devolvidos sem alteração. Não é um serializador para objetos arbitrários.
    """
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
    """Exportar tabelas, resumo JSON e as figuras fornecidas para uma pasta.

    Parameters
    ----------
    resultado : dict
        Dicionário completo retornado por analise.analisar_serie.
    pasta : Path
        Diretório de saída como pathlib.Path.
    figuras : dict[str, plt.Figure]
        Mapa nome->Figure; {} salva apenas os dados, sem gráficos.

    Returns
    -------
    None
        Grava três CSV, resumo.json e um PNG por figura fornecida.

    Notes
    -----
    Cria a pasta e seus pais; arquivos de mesmo nome são sobrescritos.
    JSON usa UTF-8, indentação 2 e null para valores reais não finitos.
    PNG usa 180 dpi e bbox_inches="tight". Não fecha as figuras.
    Erros de criação/escrita podem propagar OSError.
    """
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
