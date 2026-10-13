# Análise de curvas de luz de estrelas variáveis

O projeto estima períodos e caracteriza curvas de luz a partir de observações
AAVSO em CSV ou TXT. O fluxo combina Lomb–Scargle ponderado, inspeção da janela
espectral, ajuste de Fourier aos dados individuais, seleção por BIC,
significância de Baluev e bootstrap com tempos fixos.

O ponto de entrada é [main.py](main.py). A descrição das equações, hipóteses,
limitações e referências está em [Métodos matemáticos e estatísticos — PDF](docs/METODOS.pdf).
As funções dos módulos ativos têm docstrings com finalidade, parâmetros,
unidades, retornos e observações sobre erros e efeitos da execução.

## Estrutura do projeto

| Arquivo | Responsabilidade e funções principais |
| --- | --- |
| [main.py](main.py) | `main`: coordenar leitura, análise, impressão, gráficos e exportação; tratar erros de execução. |
| [cli.py](cli.py) | `criar_parser`, `validar_parametros`: opções e validações da linha de comando. |
| [dados.py](dados.py) | `carregar_dados`: leitura AAVSO, filtros, remoção de dados inválidos e corte robusto opcional. |
| [lomb_scargle.py](lomb_scargle.py) | `criar_lomb_scargle`, `calcular_periodograma`, `indices_picos_distintos`: modelo sinusoidal, grade e picos. |
| [janela_espectral.py](janela_espectral.py) | `calcular_janela_espectral`, `analisar_janela_espectral`, `classificar_relacao`: janela e possíveis aliases. |
| [fourier.py](fourier.py) | `AjusteFourier`, `matriz_fourier`, `avaliar_fourier`, `parametros_fourier`: ajuste ponderado e parâmetros de forma. |
| [selecao_modelos.py](selecao_modelos.py) | `refinar_frequencia`, `selecionar_candidatos`: refinamento local, ordens e escolha entre candidatos. |
| [estatistica.py](estatistica.py) | `calcular_significancia`, `bootstrap_periodos`, `resumir_bootstrap`, `binning_ponderado`. |
| [analise.py](analise.py) | `analisar_serie`: coordenar as etapas científicas e devolver o resultado completo em memória. |
| [graficos.py](graficos.py) | `criar_graficos`: curva de luz, periodograma/janela e curva dobrada. |
| [resultados.py](resultados.py) | `criar_tabela_candidatos`, `criar_tabela_janela`, `imprimir_resultados`, `valor_json`, `salvar_resultados`. |
| [../codigo_estrelas_v2.py](../codigo_estrelas_v2.py) | Versão original, localizada na pasta superior, disponível como referência executável. |
| [requirements.txt](requirements.txt) | Dependências diretas, sem fixar versões. |
| [docs/METODOS.pdf](docs/METODOS.pdf) | Equações, hipóteses, limitações e referências dos métodos usados. |

Os módulos ficam no mesmo diretório; execute os comandos a partir dele.
Não há dependências circulares. A implementação modularizada não importa a
referência original. A construção do Lomb–Scargle é compartilhada entre a análise
e o bootstrap. As docstrings foram acrescentadas aos módulos ativos; o arquivo
original continua intacto.

O PDF apresenta as equações renderizadas, sumário e referências clicáveis.
A pasta `docs/` contém apenas o PDF; a fonte Markdown e o script de geração
não estão incluídos nesta cópia do projeto.

## Instalação

O ambiente validado usa **Python 3.12**. As dependências são NumPy (arrays e
álgebra linear), pandas (leitura e tabelas), SciPy (picos e otimização), Astropy
(Lomb–Scargle e FAP) e Matplotlib (gráficos).

Em Linux/macOS, dentro da pasta do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Em Windows, usando PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

No Windows, substitua `python` nos exemplos pelo caminho
`.\.venv\Scripts\python.exe` quando o ambiente não estiver ativado.

## Dados de entrada

O arquivo precisa ter cabeçalho. As colunas numéricas obrigatórias são:

| Coluna | Conteúdo |
| --- | --- |
| `JD` | Tempo da observação em dias. |
| `Magnitude` | Magnitude observada. |
| `Uncertainty` | Incerteza de uma observação, em magnitude, estritamente positiva. |
| `Band` | Banda fotométrica; obrigatória quando há filtro de banda. |
| `Observer Code` | Código do observador; obrigatório quando há filtro de observador. |

Exemplo de estrutura (as três linhas abaixo apenas ilustram o formato):

```csv
JD,Magnitude,Uncertainty,Band,Observer Code
2459000.12,11.52,0.03,V,SAH
2459000.27,11.61,0.04,V,SAH
2459001.08,11.38,0.03,V,SAH
```

São aceitos separadores inferidos pelo pandas, incluindo vírgula, ponto e
vírgula e tabulação. Comentários começam com `#`. Nomes de colunas têm espaços
externos removidos, mas devem manter a grafia acima. Banda e observador são
comparados ignorando caixa e espaços externos. `TODOS` desativa cada filtro.

Linhas com números inválidos, valores não finitos ou incerteza não positiva
são removidas. JD repetido é preservado; as observações são ordenadas por tempo.
Não há conversão de escala de tempo, correção baricêntrica, conversão entre
fluxo e magnitude ou ajuste de diferenças de zero entre bandas/observadores.

A análise exige mais de `2 + 2*n_termos` observações após os filtros, mesmo
quando a ordem será selecionada por BIC. Com o padrão de 4 harmônicos, são
necessárias pelo menos **11 observações**, cobrindo mais de um instante.

## Execução pela linha de comando

Para analisar, imprimir os resultados e exibir gráficos:

```bash
python main.py observacoes.txt --banda V --observador SAH
```

Para salvar tabelas, resumo e gráficos, sem abrir janelas:

```bash
python main.py observacoes.txt --saida resultados
```

Para usar todos os observadores da banda V e executar sem gráficos:

```bash
python main.py observacoes.txt --observador TODOS --sem-graficos --saida resultados
```

Exemplo com intervalo de períodos explícito e bootstrap paramétrico:

```bash
python main.py observacoes.txt --periodo-minimo 0.2 --periodo-maximo 10 --n-termos 3 --bootstrap 200 --metodo-bootstrap parametrico --semente 12345 --saida resultados
```

Use `--ordem-fixa` para ajustar exatamente `--n-termos`. `--bootstrap 0` desativa
a reamostragem. `--sigma-clip 3` solicita um corte único pela mediana/MAD das
magnitudes; o corte é desativado por padrão para preservar extremos físicos.

| Argumento/opção | Padrão | Finalidade |
| --- | --- | --- |
| `arquivo` | obrigatório | Caminho do CSV/TXT. |
| `--banda` | `V` | Selecionar banda; `TODOS` desativa. |
| `--observador` | `SAH` | Selecionar observador; `TODOS` desativa. |
| `--periodo-minimo` | `0.1` | Menor período, em dias, positivo. |
| `--periodo-maximo` | `100.0` | Maior período, em dias; limitado ao intervalo observado. |
| `--n-termos` | `4` | Ordem máxima de Fourier, pelo menos 1. |
| `--ordem-fixa` | desativada | Usar exatamente a ordem informada. |
| `--amostras-por-pico` | `10` | Sobreamostragem da grade, pelo menos 5. |
| `--n-candidatos` | `15` | Máximo de picos LS examinados antes de acrescentar sub-harmônicos, pelo menos 1. |
| `--n-bins` | `50` | Intervalos de fase apenas para o gráfico, pelo menos 5. |
| `--sigma-clip` | nenhum | Multiplicador positivo da escala MAD para o corte opcional. |
| `--bootstrap` | `200` | Número de reamostragens; inteiro não negativo. |
| `--semente` | `12345` | Semente do gerador aleatório. |
| `--metodo-bootstrap` | `residuos` | `residuos` ou `parametrico`. |
| `--saida` | nenhuma | Diretório de exportação; é criado se necessário. |
| `--sem-graficos` | desativada | Desativar criação, exibição e gravação dos gráficos. |
| `-h`, `--help` | — | Mostrar ajuda e encerrar. |

O período mínimo deve ser menor que o máximo efetivo. Erros tratados encerram
a CLI com código 2; execução bem-sucedida e ajuda encerram com código 0.

## Arquivos de saída

Com `--saida`, os arquivos existentes de mesmo nome são sobrescritos.

| Arquivo | Conteúdo |
| --- | --- |
| `resumo.json` | Período selecionado, métricas, maior pico LS/FAP, amplitudes, parâmetros de forma e resumo do bootstrap. |
| `periodos_candidatos.csv` | Todos os candidatos refinados, origem, relação, métricas e indicação do selecionado. |
| `picos_janela.csv` | Frequências, períodos e potências dos picos da janela. |
| `periodos_bootstrap.csv` | Períodos de todas as réplicas válidas; somente cabeçalho se não houver nenhuma. |
| `curva_luz.png` | Magnitude em função do tempo. |
| `periodograma_janela.png` | Periodograma, níveis FAP e janela observacional. |
| `curva_fase.png` | Observações dobradas, médias por bin e modelo de Fourier. |

CSV é gravado sem índice. JSON usa UTF-8 e converte reais não finitos para
`null`; PNG usa 180 dpi. O nome histórico `semamplitude_modelo`, no dicionário,
e `semamplitude_magnitude`, no JSON, corresponde à **semiamplitude**.

Sem `--saida`, os resultados são impressos e as figuras são exibidas quando
habilitadas. Com `--sem-graficos`, os três PNG não são produzidos.

## Uso dos módulos em Python

### Fluxo completo com controle sobre os resultados

O exemplo abaixo pressupõe um arquivo válido com observações suficientes:

```python
from pathlib import Path
import matplotlib.pyplot as plt

from dados import carregar_dados
from analise import analisar_serie
from graficos import criar_graficos
from resultados import imprimir_resultados, salvar_resultados

tempo, magnitude, erro = carregar_dados(
    Path("observacoes.txt"), banda="V", observador="SAH"
)
resultado = analisar_serie(
    tempo, magnitude, erro,
    periodo_minimo=0.1, periodo_maximo=100.0,
    n_termos=4, n_bootstrap=200, semente=12345,
)
imprimir_resultados(resultado)

ajuste = resultado["melhor_ajuste"]
print(ajuste.periodo, ajuste.coeficientes, ajuste.bic)
print(resultado["tabela_candidatos"])

figuras = criar_graficos(resultado)
salvar_resultados(resultado, Path("resultados_api"), figuras)
plt.close("all")
```

`analisar_serie` valida os vetores e os parâmetros científicos, mas não aplica
os filtros da leitura. `criar_graficos` cria figuras sem salvar ou exibir.
`salvar_resultados(resultado, pasta, {})` exporta apenas CSV e JSON. Para exibir
as figuras pela API, chame `plt.show()`; seu fechamento fica a cargo do chamador.

### Uso de uma etapa isolada

Após carregar os três vetores, é possível investigar uma frequência específica:

```python
import numpy as np
from lomb_scargle import calcular_periodograma
from janela_espectral import calcular_janela_espectral
from fourier import avaliar_fourier, matriz_fourier, parametros_fourier

tempo_relativo = tempo - np.min(tempo)
ls, frequencias, potencia = calcular_periodograma(
    tempo_relativo, magnitude, erro,
    frequencia_minima=0.1, frequencia_maxima=5.0,
    amostras_por_pico=10,
)
janela = calcular_janela_espectral(tempo_relativo, frequencias)
f_pico = float(frequencias[np.argmax(potencia)])
ajuste = avaliar_fourier(tempo_relativo, magnitude, erro, f_pico, n_termos=2)
forma = parametros_fourier(ajuste.coeficientes)
modelo = matriz_fourier(tempo_relativo, ajuste.frequencia, ajuste.n_termos) @ ajuste.coeficientes
```

Esse exemplo usa a frequência do maior pico na grade. O fluxo completo também
refina frequências, compara ordens e examina sub-harmônicos, portanto pode
selecionar outro período. Funções de baixo nível pressupõem dados válidos e
alinhados; a validação completa de entrada está em `analisar_serie`.

Para consultar as assinaturas e docstrings:

```python
from selecao_modelos import selecionar_candidatos
from estatistica import bootstrap_periodos
help(selecionar_candidatos)
help(bootstrap_periodos)
```

Também é possível executar `python -m pydoc fourier` no terminal.

### Dicionário retornado por `analisar_serie`

| Chaves | Tipo, unidade e significado |
| --- | --- |
| `tempo`, `tempo_relativo`, `tempo_referencia`, `baseline` | Arrays/scalars em dias; vetores em ordem temporal, época mínima e duração. |
| `magnitude`, `erro` | Arrays em magnitude, alinhados ao tempo. |
| `frequencias`, `potencia` | Grade LS em dia⁻¹ e potência adimensional. |
| `frequencias_janela`, `potencia_janela` | Grade da janela em dia⁻¹ e sua potência. |
| `frequencia_pico_ls`, `periodo_pico_ls`, `potencia_pico_ls` | Maior pico da grade sinusoidal, em dia⁻¹, dias e potência. |
| `fap_baluev`, `niveis_fap` | Probabilidade aproximada sob a hipótese nula e mapa probabilidade → nível de potência. |
| `melhor_ajuste` | `AjusteFourier`, com frequência refinada, coeficientes, resíduos e métricas. |
| `tabela_candidatos`, `tabela_janela` | `pandas.DataFrame`; mesmas tabelas exportadas para CSV. |
| `resolucao_frequencia` | Escala `1/baseline`, em dia⁻¹. |
| `periodos_bootstrap`, `resumo_bootstrap` | Array de períodos válidos em dias e diagnóstico de recuperação/modos. |
| `fase`, `magnitude_fase`, `erro_fase` | Observações reordenadas por fase, com fases em [0, 1). |
| `fase_media`, `magnitude_media`, `erro_medio` | Médias e erros dos bins com pelo menos duas observações. |
| `fase_modelo`, `magnitude_modelo` | 1.000 amostras do modelo entre as fases 0 e 1. |
| `amplitude_pico_a_pico`, `semamplitude_modelo`, `intervalo_percentil_90` | Métricas em magnitude: amplitude do modelo, metade dela e P95−P5 dos dados. |
| `parametros_fourier`, `convencao_fase_fourier` | Mapa A1/A2/R21/phi21 e descrição da convenção de fase. |

`AjusteFourier.modelo` e `.residuos` seguem a ordem temporal, não a ordem de fase.
A primeira linha de `tabela_candidatos` é a selecionada. Pelo desempate, ela
pode ter `delta_BIC` maior que zero. A FAP refere-se ao maior pico LS, que pode
diferir do período selecionado pelo ajuste de Fourier.

## Testes e implementação original

Esta cópia do projeto não inclui a suíte `tests/`, o arquivo de versões
`requirements-testes.txt` nem o registro `VALIDACAO.md`. Portanto, os testes
de equivalência da modularização não podem ser reproduzidos apenas com os
arquivos presentes nesta pasta.

A versão original está na pasta superior. Para executá-la a partir de
`codigo_modularizado/`, use:

```bash
python ../codigo_estrelas_v2.py observacoes.txt
```

O SHA-256 do arquivo original, conferido nesta atualização do README, é:

```text
41f711f831e1d6aeb536578e66f7164472e46b15b46d6af2f06f1da4d915bf7d
```
