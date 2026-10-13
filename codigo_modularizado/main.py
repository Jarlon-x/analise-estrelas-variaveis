"""Entrada do programa. Uso: python main.py observacoes.txt [opções]."""

from __future__ import annotations

import matplotlib.pyplot as plt

from analise import analisar_serie
from cli import criar_parser, validar_parametros
from dados import carregar_dados
from graficos import criar_graficos
from resultados import imprimir_resultados, salvar_resultados


def main() -> None:
    """Executar o fluxo completo usando os argumentos presentes em sys.argv.

    Returns
    -------
    None
        Imprime resultados e, conforme as opções, salva arquivos ou exibe gráficos.

    Notes
    -----
    Sem pasta de saída, chama pyplot.show() quando há figuras. Com pasta,
    salva e fecha as figuras. Erros de arquivo, acesso, valor ou execução
    tratados encerram o programa via SystemExit(2). --help encerra com 0.
    Nenhum argumento Python é recebido diretamente.
    """
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
