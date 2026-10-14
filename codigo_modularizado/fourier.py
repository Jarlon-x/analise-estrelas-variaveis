"""Modelo, ajuste ponderado e parâmetros de forma da série de Fourier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AjusteFourier:
    """Armazenar os coeficientes e diagnósticos de um ajuste de Fourier.

    Attributes
    ----------
    frequencia : float
        Frequência em dia^-1; a propriedade periodo fornece seu inverso em dias.
    n_termos : int
        Número H de harmônicos ajustados.
    coeficientes : numpy.ndarray
        Vetor de tamanho 1 + 2*H: [c, a1, b1, ..., aH, bH], em magnitude.
    modelo : numpy.ndarray
        Magnitudes modeladas nos tempos fornecidos, vetor (N,).
    residuos : numpy.ndarray
        Magnitude observada menos o modelo, vetor (N,).
    rms_ponderado : float
        Raiz da média ponderada dos resíduos ao quadrado, em magnitude.
    chi2 : float
        Soma adimensional dos quadrados de residuos/erro.
    chi2_reduzido : float
        chi2/(N-p), com p=1+2*H; NaN se N-p não for positivo.
    bic : float
        chi2 + p*log(N), sem contar a frequência como parâmetro adicional.
    posto : int
        Posto da matriz de Fourier ponderada, retornado por lstsq.
    numero_condicao : float
        Razão entre maior e menor valor singular da matriz ponderada.
    """

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
        """Obter o período correspondente à frequência armazenada.

        Returns
        -------
        float
            Período em dias, calculado como 1/frequencia.
        """
        return 1.0 / self.frequencia


def matriz_fourier(
    tempo_relativo: np.ndarray,
    frequencia: float,
    n_termos: int,
) -> np.ndarray:
    """Construir a matriz de Fourier em uma frequência fixa.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    frequencia : float
        Frequência em ciclos por dia.
    n_termos : int
        Número de harmônicos da série de Fourier.

    Returns
    -------
    numpy.ndarray
        Matriz (N, 1 + 2*n_termos) com colunas
        [1, sin(2*pi*f*t), cos(2*pi*f*t), ..., sin(H*...), cos(H*...)].

    Notes
    -----
    A fase é reduzida módulo 1 antes da avaliação trigonométrica.
    Esta função não ajusta coeficientes nem aplica pesos.
    """
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
    """Ajustar Fourier por mínimos quadrados ponderados em frequência fixa.

    Parameters
    ----------
    tempo_relativo : np.ndarray
        Tempos em dias relativos à mesma época de referência, vetor (N,).
    magnitude : np.ndarray
        Magnitudes individuais, vetor (N,) alinhado aos tempos e erros.
    erro : np.ndarray
        Incertezas de 1 sigma em magnitude, finitas e positivas, vetor (N,).
    frequencia : float
        Frequência em ciclos por dia.
    n_termos : int
        Número de harmônicos da série de Fourier.

    Returns
    -------
    AjusteFourier
        Coeficientes, modelo nos N tempos, resíduos e métricas do ajuste.

    Notes
    -----
    Resolve o sistema escalado por 1/erro com numpy.linalg.lstsq, rcond=None.
    O BIC é chi2 + (1 + 2*n_termos)*log(N); a frequência não entra nessa
    contagem. Não há reescala dos erros nem ajuste aos dados agrupados.
    Lança ValueError para frequência não positiva/não finita e LinAlgError
    para posto incompleto, singularidade ou coeficientes/BIC não finitos.
    """
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


def parametros_fourier(coeficientes: np.ndarray) -> dict[str, float]:
    """Converter os dois primeiros harmônicos em parâmetros de forma.

    Parameters
    ----------
    coeficientes : np.ndarray
        Vetor [c, a1, b1, ..., aH, bH] do modelo em senos e cossenos.

    Returns
    -------
    dict
        A1 e A2 em magnitude, R21=A2/A1 e phi21 em radianos no intervalo
        [0, 2*pi). Valores ausentes ou fases indefinidas são numpy.nan.

    Notes
    -----
    Adota A_k*cos(2*pi*k*fase + phi_k), com phi_k=atan2(-a_k, b_k).
    Amplitudes <= 100*eps*max(1, max(abs(coeficientes))) são tratadas
    como numericamente nulas na razão/fase, conforme o ramo aplicável.
    """
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
