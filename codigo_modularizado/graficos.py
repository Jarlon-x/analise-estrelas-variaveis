"""Gráficos da curva de luz, periodograma, janela e curva dobrada."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def criar_graficos(resultado: dict) -> dict[str, plt.Figure]:
    """Gerar gráficos diagnósticos a partir de uma análise completa.

    Parameters
    ----------
    resultado : dict
        Dicionário completo retornado por analise.analisar_serie.

    Returns
    -------
    dict of str to matplotlib.figure.Figure
        Figuras com chaves curva_luz, periodograma_janela e curva_fase.

    Notes
    -----
    Cria figuras abertas, mas não as salva nem chama show(). O chamador
    controla a exibição e o fechamento. Os eixos de magnitude são invertidos.
    """
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
