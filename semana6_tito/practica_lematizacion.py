import spacy
nlp = spacy.load('es_core_news_sm')
texto = 'Los estudiantes están estudiando en las bibliotecas'
doc = nlp(texto)
for token in doc:
    print(token.text, '->', token.lemma_)
def lematizar_texto(texto):
    doc = nlp(str(texto).lower())
    lemas = []
    for token in doc:
        if token.is_alpha:
            lemas.append(token.lemma_)
    return ' '.join(lemas)
import pandas as pd
df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')
df['texto_lemma'] = df['texto'].apply(lematizar_texto)
df.to_csv(
'resultados/dataset_tito_lemma.csv',
index=False,
encoding='utf-8-sig'
)
print(df[['texto', 'texto_lemma']].head(10))
stem = pd.read_csv('resultados/dataset_tito_stem.csv')
lemma = pd.read_csv('resultados/dataset_tito_lemma.csv')
comparacion = pd.DataFrame({
    'original': df['texto'],
    'stemming': stem['texto_stem'],
    'lematizacion': lemma['texto_lemma']
})
print(comparacion.head(15))
comparacion.to_csv(
    'resultados/comparacion_stem_lemma.csv',
    index=False,
    encoding='utf-8-sig'
)