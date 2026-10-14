"""Opções de linha de comando e validação dos parâmetros."""

from __future__ import annotations

import argparse
from pathlib import Path


def criar_parser() -> argparse.ArgumentParser:
    """Construir a interface de linha de comando com os padrões originais.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado; os argumentos ainda não foram interpretados.

    Notes
    -----
    Use parse_args() para obter o Namespace aceito por validar_parametros.
    """
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
    """Verificar os limites e contagens fornecidos pela linha de comando.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace produzido por criar_parser().parse_args().

    Returns
    -------
    None
        Retorna normalmente quando os parâmetros são válidos.

    Notes
    -----
    Lança ValueError com a mensagem original se algum limite for inválido.
    Não verifica a existência nem o conteúdo do arquivo de observações.
    """
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
