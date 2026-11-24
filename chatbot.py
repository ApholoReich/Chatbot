import json
import os
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()  

DATA = Path(__file__).resolve().parent / "data"
ARCHIVO_SUBSIDIOS = DATA / "Subsidios_De_Vivienda_Asignados_20260922.csv"
MAX_FILAS = 40  # límite de filas que una herramienta devuelve al modelo



def _normalizar(texto):
    """Mayúsculas y sin tildes, para comparar nombres sin errores."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.upper().strip()


@lru_cache(maxsize=1)
def _subsidios():
    """Carga el CSV una sola vez y lo deja en memoria."""
    df = pd.read_csv(ARCHIVO_SUBSIDIOS, dtype=str)
    for col in ("Año de Asignación", "Hogares"):
        df[col] = pd.to_numeric(
            df[col].str.replace(",", "", regex=False).str.strip(), errors="coerce"
        )
    df["Valor Asignado"] = pd.to_numeric(
        df["Valor Asignado"].str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["Año de Asignación", "Hogares"]).copy()
    df["Año de Asignación"] = df["Año de Asignación"].astype(int)
    for col in ("Departamento", "Municipio", "Programa", "Estado de Postulación"):
        df[col + "_n"] = df[col].map(_normalizar)
    return df


def _coincide(serie, valor):
    """Coincidencia exacta; si no hay, coincidencia parcial (ej. 'valle')."""
    v = _normalizar(valor)
    exacto = serie == v
    return exacto if exacto.any() else serie.str.contains(v, regex=False)


def _filtrar(departamento, municipio, programa, estado, anio_inicio, anio_fin):
    df = _subsidios()
    if departamento:
        df = df[_coincide(df["Departamento_n"], departamento)]
    if municipio:
        df = df[_coincide(df["Municipio_n"], municipio)]
    if programa:
        df = df[_coincide(df["Programa_n"], programa)]
    if estado:
        df = df[_coincide(df["Estado de Postulación_n"], estado)]
    if anio_inicio:
        df = df[df["Año de Asignación"] >= anio_inicio]
    if anio_fin:
        df = df[df["Año de Asignación"] <= anio_fin]
    return df


def _json_seguro(df):
    """DataFrame -> lista de dicts sin tipos de numpy ni NaN."""
    return json.loads(df.head(MAX_FILAS).to_json(orient="records", force_ascii=False))


def _leer_resultado(nombre):
    return pd.read_csv(DATA / nombre, encoding="utf-8-sig")



def consultar_historico(
    departamento: str = "",
    municipio: str = "",
    programa: str = "",
    estado: str = "",
    anio_inicio: int = 0,
    anio_fin: int = 0,
) -> dict:
    """Suma los hogares y el valor asignado (en pesos COP) de los subsidios
    históricos, con el desglose por año. Todos los filtros son opcionales;
    usa los que mencione la pregunta (departamento, municipio, programa,
    estado de postulación o rango de años). Con anio_inicio/anio_fin en 0
    no se filtra por años."""
    df = _filtrar(departamento, municipio, programa, estado, anio_inicio, anio_fin)
    if df.empty:
        return {"error": "No hay registros con esos filtros."}
    por_anio = (
        df.groupby("Año de Asignación")
        .agg(hogares=("Hogares", "sum"), valor_cop=("Valor Asignado", "sum"))
        .reset_index()
        .rename(columns={"Año de Asignación": "anio"})
    )
    resultado = {
        "total_hogares": int(df["Hogares"].sum()),
        "total_valor_asignado_cop": int(df["Valor Asignado"].sum()),
        "por_anio": _json_seguro(por_anio),
    }
    if (por_anio["anio"] == 2026).any():
        resultado["nota"] = "2026 es un año incompleto."
    return resultado


def ranking(
    agrupar_por: str,
    top: int = 5,
    departamento: str = "",
    programa: str = "",
    estado: str = "",
    anio_inicio: int = 0,
    anio_fin: int = 0,
) -> dict:
    """Devuelve el ranking (de mayor a menor) de hogares y valor asignado.
    agrupar_por debe ser 'departamento', 'municipio' o 'programa'. Úsalo para
    preguntas como 'qué departamento tiene más subsidios' o 'los 5 programas
    más grandes'. Los demás parámetros son filtros opcionales."""
    columnas = {
        "departamento": ["Departamento"],
        "municipio": ["Municipio", "Departamento"],
        "programa": ["Programa"],
    }
    clave = _normalizar(agrupar_por).lower()
    if clave not in columnas:
        return {"error": "agrupar_por debe ser departamento, municipio o programa."}
    df = _filtrar(departamento, "", programa, estado, anio_inicio, anio_fin)
    if df.empty:
        return {"error": "No hay registros con esos filtros."}
    top = max(1, min(int(top), 20))
    tabla = (
        df.groupby(columnas[clave])
        .agg(hogares=("Hogares", "sum"), valor_cop=("Valor Asignado", "sum"))
        .reset_index()
        .sort_values("hogares", ascending=False)
        .head(top)
    )
    return {"ranking": _json_seguro(tabla)}


def listar_series_predicciones() -> dict:
    """Lista los nombres de las series disponibles en las predicciones a 10
    años (por ejemplo 'General', 'Municipio 66001' o 'Programa: MI CASA YA')."""
    df = _leer_resultado("predicciones_10_anios.csv")
    return {"series": sorted(df["Serie"].dropna().unique().tolist())}


def consultar_predicciones(serie: str) -> dict:
    """Devuelve las predicciones del modelo (proyección a 10 años) de una
    serie. Usa el nombre exacto de listar_series_predicciones. Para Pereira
    la serie es 'Municipio 66001'; para un programa es 'Programa: <nombre>'."""
    df = _leer_resultado("predicciones_10_anios.csv")
    series_n = df["Serie"].map(_normalizar)
    objetivo = _normalizar(serie)
    filtro = series_n == objetivo
    if not filtro.any():
        filtro = series_n.str.contains(objetivo, regex=False)
    if not filtro.any():
        return {"error": "Serie no encontrada. Usa listar_series_predicciones."}
    return {"predicciones": _json_seguro(df[filtro])}


def consultar_evaluacion() -> dict:
    """Devuelve el resumen de la evaluación del modelo de predicción
    (qué tan bien o mal predice). Úsalo para preguntas sobre la calidad,
    el error o la confiabilidad del modelo."""
    return {"evaluacion": _json_seguro(_leer_resultado("evaluacion_resumen.csv"))}


HERRAMIENTAS = [
    consultar_historico,
    ranking,
    listar_series_predicciones,
    consultar_predicciones,
    consultar_evaluacion,
]


#conexion gemini
def _instrucciones():
    df = _subsidios()
    programas = ", ".join(sorted(df["Programa"].dropna().unique()))
    return f"""Eres el asistente del proyecto "Predicciones de subsidios de vivienda" de Colombia.
Respondes siempre en español, de forma clara y breve.

Reglas:
- Usa SIEMPRE las herramientas para obtener cifras. Nunca inventes números.
- Si la pregunta no trata de subsidios de vivienda o de este proyecto, di amablemente que solo puedes ayudar con ese tema.
- Los datos históricos llegan hasta 2025; 2026 está incompleto y se excluyó del entrenamiento.
- El modelo de predicción es una regresión lineal sobre la suma de hogares por año.
- En las predicciones, Pereira corresponde a la serie 'Municipio 66001' y cada programa se llama 'Programa: <nombre>'.
- Los valores de dinero están en pesos colombianos (COP). Escribe los números con separador de miles.
- Si una herramienta no encuentra datos, dilo con claridad y sugiere reformular la pregunta.
- Si el usuario pregunta por qué una predicción es poco confiable, usa la evaluación del modelo.

Programas existentes: {programas}."""


@lru_cache(maxsize=1)
def _cliente():
    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    return genai.Client(api_key=clave)


def _generar(modelo, contenidos, config):
    """Llama a Gemini. Si el modelo está saturado (error 503/5xx), reintenta
    3 veces y luego prueba el modelo de respaldo GEMINI_MODEL_RESPALDO."""
    modelos = [modelo]
    respaldo = os.getenv("GEMINI_MODEL_RESPALDO")
    if respaldo and respaldo != modelo:
        modelos.append(respaldo)

    ultimo_error = None
    for nombre in modelos:
        for intento in range(3):
            try:
                return _cliente().models.generate_content(
                    model=nombre, contents=contenidos, config=config
                )
            except errors.ServerError as error:  # 5xx: saturación temporal
                ultimo_error = error
                if intento < 2:
                    time.sleep(2 ** intento)  # espera 1 s y luego 2 s
    raise ultimo_error


def responder(pregunta, historial=None):
    """Envía la pregunta (y los últimos mensajes) a Gemini y devuelve el texto."""
    modelo = os.getenv("GEMINI_MODEL")
    if not modelo:
        raise RuntimeError("Falta GEMINI_MODEL en el archivo .env")

    contenidos = []
    for turno in (historial or [])[-6:]:
        if not isinstance(turno, dict):
            continue
        rol = "user" if turno.get("rol") == "user" else "model"
        texto = str(turno.get("texto", ""))[:1500]
        if texto:
            contenidos.append(types.Content(role=rol, parts=[types.Part(text=texto)]))
    while contenidos and contenidos[0].role != "user":
        contenidos.pop(0)  # el historial debe empezar con un mensaje del usuario
    contenidos.append(types.Content(role="user", parts=[types.Part(text=pregunta)]))

    config = types.GenerateContentConfig(
        system_instruction=_instrucciones(),
        tools=HERRAMIENTAS,  # Gemini llama las funciones automáticamente
        temperature=0.2,
    )
    respuesta = _generar(modelo, contenidos, config)
    return (respuesta.text or "").strip() or "No pude generar una respuesta. Intenta reformular la pregunta."