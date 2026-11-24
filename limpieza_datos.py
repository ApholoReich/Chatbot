import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


print("Cargando el archivo CSV...")
df = pd.read_csv('Subsidios_De_Vivienda_Asignados_20260922.csv')

print("\nPrimeras filas:")
print(df.head())

print(df.columns.tolist())

print("\nColumnas disponibles:")
print(list(df.columns))

# Limpieza y conversion de los años de asignacion a dato numerico
df['Año de Asignación'] = (
    df['Año de Asignación']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df['Año de Asignación'] = pd.to_numeric(
    df['Año de Asignación'],
    errors='coerce'
)

#  Limpieza de la columna Hogares
df['Hogares'] = (
    df['Hogares']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['Hogares'] = pd.to_numeric(
    df['Hogares'],
    errors='coerce'
)

# Eliminar filas con datos no validos
df = df.dropna(
    subset=['Año de Asignación', 'Hogares']
)

# Agrupar y sumar los hogares por año
df_agrupado = (
    df.groupby('Año de Asignación', as_index=False)['Hogares']
    .sum()
    .sort_values('Año de Asignación')
)

print("\nHogares por año:")
print(df_agrupado)

# Excluir 2026 por ser un año incompleto

df_entrenamiento = df_agrupado[df_agrupado['Año de Asignación'] < 2026].copy()

if len(df_entrenamiento) < 2:
    print(
        '\nNo hay suficientes años historicos para entrenar'
    )
else:
    # Preparar los datos
    X = df_entrenamiento[['Año de Asignación']]
    y = df_entrenamiento['Hogares']

    # Entrenamiento del modelo
    modelo = LinearRegression()
    modelo.fit(X, y)
 
    print("\nModelo entrenado correctamente.")

    # Prediccion para 2027
    año_futuro = pd.DataFrame({
        'Año de Asignación': [2027]
    })

    prediccion = modelo.predict(año_futuro)[0]

    print(
        f"Predicción para 2027: {prediccion:,.0f} hogares"
    )

# Grafica de los resultados generales
plt.figure(figsize=(10, 6))

plt.scatter(
        X['Año de Asignación'],
        y,
        label='Datos históricos'
    )

plt.plot(
        X['Año de Asignación'],
        modelo.predict(X),
        color='red',
        label='Tendencia del modelo'
    )

plt.scatter(
        [2027],
        [prediccion],
        color='green',
        s=100,
        label='Predicción 2027'
    )

plt.xlabel('Año de asignación')
plt.ylabel('Número de hogares')
plt.title('Predicción de subsidios de vivienda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Próximos 10 años
ultimo_anio = X.max().max()

X_futuro = np.arange(
    int(ultimo_anio) + 1,
    int(ultimo_anio) + 11
).reshape(-1, 1)



# Generar los próximos 10 años
X_futuro = pd.DataFrame({
        'Año de Asignación': np.arange(
            ultimo_anio + 1,
            ultimo_anio + 11
        )
    })

# Prediccion de los proximos 10 años futuros
predicciones = modelo.predict(X_futuro)

# Mostrar resultados
resultados = X_futuro.copy()
resultados['Hogares estimados'] = np.round(
    predicciones
).astype(int)

print("\nPredicciones para los próximos 10 años:")
print(resultados.to_string(index=False))

# Grafica de resultados proyeccion a 10 años
plt.figure(figsize=(12, 6))

plt.scatter(
    X['Año de Asignación'],
    y,
    label='Datos históricos'
)

plt.plot(
    X['Año de Asignación'],
    modelo.predict(X),
    color='red',
    label='Tendencia histórica'
)

plt.plot(
    X_futuro['Año de Asignación'],
    predicciones,
    color='green',
    marker='o',
    linestyle='--',
    label='Predicciones futuras')

plt.xlabel('Año de asignación')
plt.ylabel('Número de hogares')
plt.title('Proyección general a 10 años subsidios de vivienda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Prediccion de hogares por municipio

df['Municipio'] = (
    df['Municipio']
    .str.normalize('NFKD')
    .str.replace(r"[\u0300-\u036f]", "", regex=True)
)
municipio = 'PEREIRA'

# Filtrar el municipio
datos_municipio = df[
    df['Municipio'].astype(str).str.strip().str.upper() == municipio
].copy()

# Agrupar los hogares por año
datos_municipio = (
    datos_municipio
    .groupby('Año de Asignación', as_index=False)['Hogares']
    .sum()
    .sort_values('Año de Asignación')
)

# Excluir 2026 por ser un año incompleto

datos_municipio = datos_municipio[
    datos_municipio['Año de Asignación'] < 2026
]

print(f"\nDatos históricos de {municipio}:")
print(datos_municipio.to_string(index=False))

if len(datos_municipio) < 2:
    print("No hay suficientes años para entrenar el modelo.")
else:
    # Preparar los datos
    X_municipio = datos_municipio[['Año de Asignación']]
    y_municipio = datos_municipio['Hogares']

    # Entrenar el modelo
    modelo_municipio = LinearRegression()
    modelo_municipio.fit(X_municipio, y_municipio)

    # Predecir 2027
    año_futuro_municipio = pd.DataFrame({
        'Año de Asignación': [2027]
    })

    prediccion = modelo_municipio.predict(
        año_futuro_municipio
    )[0]

    print(
        f"\nSubsidios de vivienda estimados para {municipio} en 2027: "
        f"{prediccion:.0f}"
    )

    if prediccion < 0:
        print(
            "No hay relacion entre las variables")


 # Agrupar hogares por programa y año
datos_programa = df.groupby(
    ['Programa', 'Año de Asignación'],
    as_index=False
)['Hogares'].sum()

# Programas disponibles
programas = datos_programa['Programa'].dropna().unique()

resultados_programas = []

for programa in programas:

    datos_actuales = datos_programa[
        datos_programa['Programa'] == programa
    ]

    # Usar únicamente años anteriores a 2026
    datos_actuales = datos_actuales[
        datos_actuales['Año de Asignación'] < 2026
    ]

    # Verificar que existan al menos dos años
    if len(datos_actuales) < 2:
        continue

    # Preparacion de los datos
    X = datos_actuales[['Año de Asignación']]
    y = datos_actuales['Hogares']

    # Entrenar el modelo
    modelo_general = LinearRegression()
    modelo_general.fit(X, y)

    # Predecir hogares para 2027
    prediccion = modelo_general.predict(
        pd.DataFrame({'Año de Asignación': [2027]})
    )[0]

    resultados_programas.append({
        'Programa': programa,
        'Hogares estimados 2027': round(max(0, prediccion))
    })

# Mostrar resultados 
resultado = pd.DataFrame(resultados_programas)

if not resultado.empty:
    resultado = resultado.sort_values(
        'Hogares estimados 2027',
        ascending=False
    )

    print('\nPredicción de hogares por programa para 2027:')
    print(resultado.to_string(index=False))
else:
    print('No hay suficientes datos históricos por programa.')


# Graficas de resultados por programa

plt.figure(figsize=(12, 8))

plt.barh(
    resultado['Programa'],
    resultado['Hogares estimados 2027']
)

plt.xlabel('Hogares estimados')
plt.ylabel('Programa')
plt.title('Predicción de hogares por programa para 2027')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()