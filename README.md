# Análise de Estrelas Variáveis

Este projeto contém ferramentas em Python para a análise de curvas de luz de estrelas variáveis.

## 🚀 Funcionalidades
* Análise de séries temporais com período Lomb-Scargle
* Decomposição de Fourier para modelagem de curvas
* Análise estatística robusta via Bootstrap

## 📦 Como Usar
A versão modularizada está em [`codigo_modularizado/`](codigo_modularizado/).
Com Python 3.12 instalado, execute em Linux/macOS:

```bash
cd codigo_modularizado
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py observacoes.txt --banda V --observador SAH --saida resultados
```

Substitua `observacoes.txt` pelo caminho do seu arquivo de observações.
O [README da versão modularizada](codigo_modularizado/README.md) apresenta
as instruções para Windows, o formato dos dados e todas as opções de execução.
Consulte também os [métodos matemáticos e estatísticos](codigo_modularizado/docs/METODOS.pdf).

A versão original permanece disponível na raiz:

```bash
python codigo_estrelas_v2.py observacoes.txt
```

## Guia sobre o código de Análise de Estrelas Variáveis
https://colab.research.google.com/drive/1h2RK7LjTbZ7QxqUaedqniC_OUnwXE-Jr?usp=sharing
