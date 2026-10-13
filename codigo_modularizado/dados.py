"""Leitura, filtragem e ordenação das observações AAVSO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def carregar_dados(
    arquivo: Path,
    banda: str = "V",
    observador: str = "SAH",
    sigma_clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ler observações AAVSO, aplicar filtros e ordenar as medidas por JD.

    Parameters
    ----------
    arquivo : Path
        Caminho pathlib.Path para um CSV ou TXT com cabeçalho.
    banda : str, default='V'
        Valor de Band, ignorando caixa e espaços externos; TODOS desativa o filtro.
    observador : str, default='SAH'
        Valor de Observer Code; TODOS desativa o filtro.
    sigma_clip : float | None, default=None
        Multiplicador positivo da escala 1.4826*MAD; None desativa o corte.

    Returns
    -------
    tuple of numpy.ndarray
        (tempo, magnitude, erro), vetores float de mesmo tamanho, ordenados
        pelo tempo. Tempos repetidos permanecem como medidas independentes.

    Notes
    -----
    Reconhece separadores por inferência do pandas e comentários iniciados
    por #. Converte valores não numéricos em inválidos e descarta linhas
    não finitas ou com incerteza não positiva. Imprime contagens no terminal.
    O corte opcional usa uma única passagem de mediana/MAD sobre magnitudes.
    Lança FileNotFoundError para caminho ausente e ValueError para erros de
    interpretação, colunas ausentes ou ausência de medidas válidas antes do
    corte; erros de acesso ao arquivo podem propagar OSError.
    """
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
