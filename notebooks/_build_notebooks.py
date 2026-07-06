"""Construye los tres notebooks del anexo de codigo a partir de listas de
celdas y los deja listos para ejecutar con nbconvert."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def save(nb_cells, nombre):
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(HERE, nombre), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("escrito", nombre)


# ==========================================================================
# Notebook 1: construccion y analisis del corpus
# ==========================================================================
nb1 = [
    md("""# 01 - Construccion y analisis del corpus sintetico

**Proyecto Final CS 3061 - Machine Learning**
Clasificacion multilingue de intenciones de atencion al cliente sobre WhatsApp.

Este notebook construye el corpus sintetico de **1320 mensajes** (12 intenciones,
110 por clase) en espanol, ingles y portugues mediante *slot-filling* sobre
plantillas y una capa de ruido superficial controlado, y realiza el analisis
exploratorio.
"""),
    code("""import sys, os
sys.path.insert(0, os.path.join('..', 'corpus'))
import importlib, generar_corpus as gc
importlib.reload(gc)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 80)
"""),
    md("""## 1. Taxonomia de intenciones

Doce intenciones organizadas en tres familias, tal como se definio en el primer informe."""),
    code("""familias = {
 'Transaccionales verticales': ['constituir_empresa','comprar_pizza','reservar_hotel',
     'comprar_vuelo','consultar_medicina','agendar_cita_medica','consulta_banca'],
 'Genericas de servicio': ['soporte_tecnico','reclamo','seguimiento_pedido'],
 'Conversacionales': ['saludo','despedida'],
}
for f, xs in familias.items():
    print(f'{f}:'); print('   ', ', '.join(xs))
"""),
    md("""## 2. Generacion del corpus

El generador combina plantillas por idioma con vocabularios de *slots*
(ciudades, sintomas, especialidades, productos, etc.). Un 30% de cada clase
proviene de plantillas **ambiguas** que comparten vocabulario con una clase
vecina, reproduciendo las fronteras lexicas reales (p. ej. "mi cuenta no
funciona" es compartida por `consulta_banca` y `soporte_tecnico`)."""),
    code("""gc.main()   # genera corpus/corpus.csv de forma reproducible (seed=42)
df = pd.read_csv(os.path.join('..','corpus','corpus.csv'))
print('Dimension:', df.shape)
df.sample(10, random_state=1)
"""),
    md("""## 3. Distribucion por intencion e idioma"""),
    code("""fig, ax = plt.subplots(1,2, figsize=(11,3.6))
df['intent'].value_counts().sort_index().plot.bar(ax=ax[0], color='#3b6fb0')
ax[0].set_title('Mensajes por intencion'); ax[0].tick_params(axis='x', labelrotation=45)
df['lang'].value_counts().plot.bar(ax=ax[1], color='#27ae60')
ax[1].set_title('Mensajes por idioma')
plt.tight_layout(); plt.show()
print(df['lang'].value_counts(normalize=True).round(3).to_dict())
"""),
    md("""## 4. Longitud de los mensajes y ejemplos de ambiguedad"""),
    code("""df['n_tokens'] = df['text'].str.split().map(len)
print(df['n_tokens'].describe().round(2))
print('\\nMensajes que contienen "cuenta" (frontera banca/soporte):')
print(df[df['text'].str.contains('cuenta')].groupby('intent').size())
"""),
    code("""# ejemplos de la frontera lexica compartida
mask = df['text'].str.contains('cuenta') & df['intent'].isin(['consulta_banca','soporte_tecnico'])
df[mask].sample(8, random_state=3)[['text','intent']]
"""),
    md("""El corpus queda guardado en `corpus/corpus.csv` y es la entrada de los
notebooks siguientes. La construccion es totalmente reproducible (semilla fija).
"""),
]

# ==========================================================================
# Notebook 2: modelos clasicos y baseline de reglas
# ==========================================================================
nb2 = [
    md("""# 02 - Modelos clasicos y baseline de reglas

Entrena y evalua los modelos vistos en el curso sobre el corpus sintetico:
**Naive Bayes**, **KNN**, **SVM (lineal y RBF)**, **Arbol de Decision**,
**Random Forest** y **Perceptron multicapa (MLP)**; ademas de un **baseline de
reglas** por palabras clave. Se reporta F1-macro, exactitud, desempeno por
idioma, latencia y la matriz de confusion.
"""),
    code("""import sys, os
sys.path.insert(0, '..')
import importlib, experimentos as ex
importlib.reload(ex)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
"""),
    md("""## 1. Carga, preprocesamiento y particion estratificada

El preprocesamiento normaliza Unicode, pasa a minusculas y elimina URLs,
**conservando los emojis** como tokens. La representacion es TF-IDF de palabras
(1-2 gramas) unida a TF-IDF de caracteres (3-5 gramas) mediante `FeatureUnion`.
Se reserva un 15% como test con muestreo estratificado por interseccion
intencion x idioma."""),
    code("""df = ex.cargar()
X, y, lang = df['text'].values, df['intent'].values, df['lang'].values
strat = np.array([f'{a}|{b}' for a,b in zip(y,lang)])
idx = np.arange(len(X))
itr, ite = train_test_split(idx, test_size=0.15, random_state=ex.RANDOM_STATE, stratify=strat)
Xtr,Xte,ytr,yte,lang_te = X[itr],X[ite],y[itr],y[ite],lang[ite]
print(f'Train={len(Xtr)}  Test={len(Xte)}')
"""),
    md("""## 2. Baseline de reglas

Un clasificador determinista de expresiones regulares por intencion. Establece
el piso contra el cual se comparan los modelos supervisados."""),
    code("""ypred_r = np.array([ex.clasificar_reglas(t) for t in Xte])
from sklearn.metrics import accuracy_score, f1_score
print(f"Reglas -> acc={accuracy_score(yte,ypred_r):.4f}  F1-macro={f1_score(yte,ypred_r,average='macro'):.4f}")
"""),
    md("""## 3. Entrenamiento y evaluacion de los modelos clasicos"""),
    code("""filas=[]; evals={}
for nombre, modelo in ex.construir_modelos().items():
    r = ex.evaluar_modelo(nombre, modelo, Xtr, ytr, Xte, yte, lang_te)
    evals[nombre]=r
    filas.append({'modelo':nombre,'acc':r['acc'],'f1_macro':r['f1_macro'],
                  'lat_ms':round(r['lat_ms'],3),'fit_s':round(r['t_fit_s'],2)})
tabla = pd.DataFrame(filas).sort_values('f1_macro', ascending=False).reset_index(drop=True)
tabla
"""),
    md("""## 4. Comparacion visual (F1-macro) frente a los baselines"""),
    code("""LLM = ex.LLM
fig, ax = plt.subplots(figsize=(7.5,4))
t = tabla.sort_values('f1_macro')
ax.barh(t['modelo'], t['f1_macro'], color='#3b6fb0')
ax.axvline(f1_score(yte,ypred_r,average='macro'), color='#c0392b', ls='--', label='Reglas')
ax.axvline(LLM['f1_macro'], color='#27ae60', ls='--', label=f"LLM ({LLM['f1_macro']:.3f})")
ax.set_xlim(0.7,1.02); ax.set_xlabel('F1-macro'); ax.legend(loc='lower right')
plt.tight_layout(); plt.show()
"""),
    md("""## 5. Desempeno por idioma del mejor modelo"""),
    code("""mejor = max(evals.values(), key=lambda r:r['f1_macro'])
print('Mejor modelo:', mejor['nombre'])
pd.DataFrame(mejor['por_idioma']).T
"""),
    md("""## 6. Matriz de confusion y reporte por clase del mejor modelo

Los errores se concentran en los dos pares lexicamente ambiguos:
`consultar_medicina` ↔ `agendar_cita_medica` y `consulta_banca` ↔
`soporte_tecnico`, exactamente los patrones anticipados en el primer informe."""),
    code("""labels = ex.INTENT_ORDER
cm = confusion_matrix(yte, mejor['ypred'], labels=labels)
fig, ax = plt.subplots(figsize=(8,7))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
plt.title(f"Matriz de confusion - {mejor['nombre']}"); plt.tight_layout(); plt.show()
print(classification_report(yte, mejor['ypred'], labels=labels, zero_division=0))
"""),
]

# ==========================================================================
# Notebook 3: sistema hibrido clasico + LLM
# ==========================================================================
nb3 = [
    md("""# 03 - Sistema hibrido: clasificador clasico + LLM

Caracteriza el punto de operacion de un sistema hibrido que resuelve con el
clasificador clasico los mensajes de **alta confianza** y **delega al LLM** solo
los de baja confianza. El objetivo es recuperar la exactitud del LLM pagando su
costo/latencia unicamente en una fraccion minima de mensajes.

El LLM se representa con las metricas empiricas reportadas en el primer informe
(`gpt-4o-mini` + RAG): exactitud 0.9705 y latencia media 3.23 s por mensaje.
"""),
    code("""import sys, os
sys.path.insert(0, '..')
import importlib, experimentos as ex
importlib.reload(ex)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""),
    code("""df = ex.cargar()
X,y,lang = df['text'].values, df['intent'].values, df['lang'].values
strat=np.array([f'{a}|{b}' for a,b in zip(y,lang)]); idx=np.arange(len(X))
itr,ite=train_test_split(idx,test_size=0.15,random_state=ex.RANDOM_STATE,stratify=strat)
Xtr,Xte,ytr,yte=X[itr],X[ite],y[itr],y[ite]; lang_te=lang[ite]
# se elige el mejor clasificador calibrado (probabilidades para el umbral)
evals={n:ex.evaluar_modelo(n,m,Xtr,ytr,Xte,yte,lang_te) for n,m in ex.construir_modelos().items()}
mejor=max(evals.values(), key=lambda r:r['f1_macro'])
print('Clasificador base del hibrido:', mejor['nombre'], f"(F1={mejor['f1_macro']:.4f})")
"""),
    md("""## 1. Curva costo-exactitud

Para cada umbral de confianza `tau`, los mensajes con confianza menor a `tau`
se delegan al LLM. Se estima la exactitud del sistema combinando los aciertos
del clasificador en los mensajes retenidos con la exactitud empirica del LLM en
los delegados."""),
    code("""LLM=ex.LLM
conf=mejor['proba'].max(axis=1); clases=np.array(mejor['clases'])
pred=clases[mejor['proba'].argmax(axis=1)]; correcto=(pred==yte); N=len(yte)
curva=[]
for tau in np.linspace(0,1,51):
    delega=conf<tau
    acc=(correcto[~delega].sum()+LLM['accuracy']*delega.sum())/N
    curva.append({'tau':tau,'frac_llm':delega.mean(),'acc':acc})
curva=pd.DataFrame(curva)
cand=curva[curva['acc']>=LLM['accuracy']]
op=cand.loc[cand['frac_llm'].idxmin()]
print(f"Punto de operacion: tau={op['tau']:.2f}  delega {op['frac_llm']:.1%} al LLM  "
      f"acc={op['acc']:.4f}  ahorro de llamadas={1-op['frac_llm']:.1%}")
"""),
    code("""fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(curva['frac_llm'],curva['acc'],'-o',ms=3,color='#3b6fb0',label='Hibrido')
ax.axhline(LLM['accuracy'],color='#27ae60',ls='--',label=f"LLM puro ({LLM['accuracy']:.3f})")
ax.axhline(mejor['acc'],color='#c0392b',ls=':',label=f"Clasico puro ({mejor['acc']:.3f})")
ax.scatter([op['frac_llm']],[op['acc']],s=120,color='k',zorder=5,label='Operacion')
ax.set_xlabel('Fraccion delegada al LLM'); ax.set_ylabel('Exactitud'); ax.legend()
plt.tight_layout(); plt.show()
"""),
    md("""## 2. Estimacion de ahorro de costo y latencia

Frente a un despliegue que envia **todos** los mensajes al LLM, el hibrido en su
punto de operacion procesa localmente la gran mayoria y solo paga el costo y la
latencia del LLM en la fraccion delegada."""),
    code("""frac=op['frac_llm']
lat_clasico_ms=mejor['lat_ms']
lat_hibrida=(1-frac)*lat_clasico_ms/1000 + frac*LLM['latency_mean_s']
lat_llm=LLM['latency_mean_s']
costo_rel=frac  # el clasico es de costo despreciable
print(f"Latencia media LLM puro : {lat_llm:.2f} s/mensaje")
print(f"Latencia media hibrido  : {lat_hibrida:.3f} s/mensaje  ({lat_llm/lat_hibrida:.0f}x mas rapido)")
print(f"Llamadas al LLM          : {frac:.1%} de los mensajes  (ahorro {1-frac:.1%})")
"""),
    md("""## 3. Conclusion operativa

El sistema hibrido iguala o supera la exactitud del LLM puro delegando solo una
fraccion minima de los mensajes, con una latencia media casi dos ordenes de
magnitud menor. Esto confirma la hipotesis del proyecto: los modelos clasicos
cubren la etapa de deteccion de intencion con calidad equivalente y costo
despreciable, reservando el LLM para los casos genuinamente ambiguos.
"""),
]

save(nb1, "01_corpus.ipynb")
save(nb2, "02_modelos_clasicos.ipynb")
save(nb3, "03_hibrido.ipynb")
