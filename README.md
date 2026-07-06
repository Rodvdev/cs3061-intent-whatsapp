# Proyecto Final CS 3061 — Clasificación Multilingüe de Intenciones (WhatsApp)

Repositorio: https://github.com/Rodvdev/cs3061-intent-whatsapp

Entrega final del curso de Machine Learning (UTEC). Compara modelos clásicos de
aprendizaje automático contra un baseline de reglas y el baseline LLM del primer
informe, sobre un corpus sintético multilingüe de atención al cliente.

## Estructura

```
proyecto-final/
├── corpus/
│   ├── generar_corpus.py     # genera el corpus (seed=42, reproducible)
│   └── corpus.csv            # 1320 mensajes · 12 intenciones · ES/EN/PT
├── experimentos.py           # pipeline: TF-IDF + 7 modelos + reglas + híbrido
├── resultados.json           # métricas volcadas
├── _img/                     # figuras generadas
├── notebooks/                # ANEXO DE CÓDIGO (ejecutables)
│   ├── 01_corpus.ipynb
│   ├── 02_modelos_clasicos.ipynb
│   └── 03_hibrido.ipynb
└── latex/                    # PAPER IEEE (Overleaf)
    ├── main.tex
    ├── refs.bib              # 25 referencias (15 nuevas + 10 del P1)
    └── _img/                 # figuras para el paper
```

## Reproducir

```bash
python corpus/generar_corpus.py   # regenera corpus.csv
python experimentos.py            # regenera métricas y figuras
```

Los notebooks se ejecutan con Jupyter y reutilizan `experimentos.py`.

## Compilar el paper

Subir la carpeta `latex/` a Overleaf (plantilla IEEE Transactions) y compilar
`main.tex` con pdfLaTeX + BibTeX. Localmente: `tectonic latex/main.tex`.

> **Pendiente:** reemplazar el enlace de repositorio placeholder en `main.tex`
> (sección "Código implementado") por el repo git o Colab real.

## Resultados clave

| Sistema | F1-macro | Latencia |
|---|---|---|
| Baseline de reglas | 0.825 | <0.01 ms |
| Mejor clásico (Naive Bayes) | **0.965** | 0.025 ms |
| Baseline LLM (P1) | 0.973 | 3230 ms |
| **Híbrido** (delega 3% al LLM) | **0.974** | ~100 ms |

El error se concentra en los dos pares léxicamente ambiguos
(`consultar_medicina`↔`agendar_cita_medica`, `consulta_banca`↔`soporte_tecnico`),
reproduciendo las confusiones del baseline LLM.
