import os
import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('spanish')

palabras = ['estudiando', 'estudiantes', 'bibliotecas', 'solicitudes']
for palabra in palabras:
    print(f"{palabra} -> {stemmer.stem(palabra)}")

print("\n" + "="*50 + "\n")

def aplicar_stemming(tokens):
    """Recibe una lista de tokens y devuelve cada token reducido a su raíz."""
    return [stemmer.stem(token) for token in tokens]

def procesar_texto(texto):
    """Limpia el texto (minúsculas, quita caracteres especiales) y aplica stemming."""
    texto = texto.lower().strip()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    tokens = texto.split()
    return [stemmer.stem(t) for t in tokens]

ejemplo = '¿Cómo solicito una beca escolar?'
print(f"Original: {ejemplo}")
print(f"Stemmed:  {procesar_texto(ejemplo)}")

print("\n" + "="*50 + "\n")

if not os.path.exists('resultados'):
    os.makedirs('resultados')

try:
    df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')

    df['texto_stem'] = df['texto'].apply(
        lambda x: ' '.join(procesar_texto(str(x)))
    )

    archivo_salida = 'resultados/dataset_tito_stem.csv'
    df.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
    print(f"Archivo guardado exitosamente en: {archivo_salida}\n")

    print("--- Muestra de los primeros 10 resultados ---")
    print(df[['texto', 'texto_stem']].head(10))

except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset_tito_v1.csv' en la carpeta actual.")