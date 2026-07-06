"""Pipeline de experimentos del proyecto final (CS 3061).

Entrena y evalua los modelos clasicos vistos en el curso sobre el corpus
sintetico multilingue, ademas de un baseline de reglas, y caracteriza un
sistema hibrido clasico + LLM. Produce metricas reales, figuras y un
volcado JSON reutilizable por el informe.

Baseline LLM: se reutilizan las metricas empiricas reportadas en el primer
informe (openai/gpt-4o-mini + RAG), pues no se dispone de acceso a la API en
este entorno. Numeros tomados de P1: accuracy 0.9705, F1-macro 0.9728,
latencia media 3.23 s.

Salida:  _img/*.png  y  resultados.json
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "_img")
os.makedirs(IMG, exist_ok=True)

# Metricas del baseline LLM tomadas del primer informe (P1)
LLM = {
    "accuracy": 0.9705, "f1_macro": 0.9728, "latency_mean_s": 3.23,
    "por_idioma": {"es": {"acc": 0.9713, "f1": 0.9718},
                    "en": {"acc": 0.9549, "f1": 0.9621},
                    "pt": {"acc": 1.0, "f1": 1.0}},
    "costo_usd_1k": 0.60,   # estimado gpt-4o-mini (~600 tok salida + RAG)
}

INTENT_ORDER = [
    "constituir_empresa", "comprar_pizza", "reservar_hotel", "comprar_vuelo",
    "consultar_medicina", "agendar_cita_medica", "consulta_banca",
    "soporte_tecnico", "reclamo", "seguimiento_pedido", "saludo", "despedida",
]

# --------------------------------------------------------------------------
# 1. Datos y preprocesamiento
# --------------------------------------------------------------------------
import unicodedata


def limpiar(texto):
    """Normalizacion Unicode y limpieza ligera, conservando emojis."""
    t = unicodedata.normalize("NFC", str(texto))
    t = t.strip().lower()
    # eliminar urls (defensivo; el corpus sintetico no las incluye)
    import re
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def construir_vectorizador():
    """TF-IDF de palabras (1-2) unido a TF-IDF de caracteres (3-5)."""
    palabras = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             min_df=2, sublinear_tf=True)
    return FeatureUnion([("word", palabras), ("char", chars)])


def cargar():
    df = pd.read_csv(os.path.join(HERE, "corpus", "corpus.csv"))
    df["text"] = df["text"].map(limpiar)
    return df


# --------------------------------------------------------------------------
# 2. Baseline de reglas (palabras clave / regex por intencion)
# --------------------------------------------------------------------------
import re

REGLAS = [
    ("constituir_empresa", r"constitu|formaliz|registr.*empresa|sac\b|ruc|incorporat|abrir.*(empresa|negocio)|set up|open.*(company|business)|abrir empresa"),
    ("comprar_pizza", r"pizza"),
    ("reservar_hotel", r"hotel|habitaci|hostal|hostel|resort|quarto|room|reserv.*(noche|night|noite)"),
    ("comprar_vuelo", r"vuelo|pasaje|flight|voo|passagem|boleto.*(avion|ciudad)|aerolinea|latam|avianca"),
    ("agendar_cita_medica", r"cita|agendar|appointment|consulta de (cardio|pedia|derma|trauma|gineco|odonto|oftalmo|ortope|dentis)|marcar.*(horario|consulta)|schedule.*appointment"),
    ("consultar_medicina", r"medicina|medicament|remedio|pastilla|dosis|paracetamol|ibuprofen|amoxicil|sintoma|dolor|fiebre|tos|gripe|febre|headache|fever|cough|medicine|take for"),
    ("consulta_banca", r"saldo|cuenta.*(ahorro|banc|credito)|tarjeta|transferir|estado de cuenta|banco|conta|cartao|balance|account.*(balance|statement|transaction)|savings"),
    ("soporte_tecnico", r"app|aplicaci|iniciar sesion|log in|login|error|contrasena|password|no funciona|nao funciona|crash|se cierra|codigo de verificacion|verification code|lento|slow"),
    ("reclamo", r"reclamo|reclama|queja|complaint|molesto|pesimo|danad|roto|defectuoso|devoluci|reembolso|refund|damaged|broken|upset|overcharg|reclamacao"),
    ("seguimiento_pedido", r"pedido|orden|order|rastrear|track|envio|paquete|package|shipment|tracking|donde esta mi|where is my|onde esta"),
    ("saludo", r"^\s*(hola|buenos dias|buenas|good morning|good afternoon|good evening|hello|hi|hey|ola|oi|boa tarde|boa noite|bom dia|saludos|que tal)"),
    ("despedida", r"gracias.*(chau|adios|bye|luego|pronto)|hasta luego|hasta pronto|nos vemos|adios|goodbye|see you|thank you.*bye|obrigado.*(tchau|adeus|logo)|ate (logo|mais|a proxima)|cuidate|que tenga buen dia"),
]


def clasificar_reglas(texto):
    for intent, patron in REGLAS:
        if re.search(patron, texto):
            return intent
    return "soporte_tecnico"   # categoria por defecto (residual)


# --------------------------------------------------------------------------
# 3. Modelos clasicos
# --------------------------------------------------------------------------
def construir_modelos():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance",
                                     metric="cosine"),
        "SVM lineal": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=3),
        "SVM RBF": SVC(C=10, gamma="scale", kernel="rbf",
                        probability=True, random_state=RANDOM_STATE),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=None, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                              early_stopping=True, random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------
# 4. Evaluacion
# --------------------------------------------------------------------------
def evaluar_modelo(nombre, modelo, Xtr, ytr, Xte, yte, lang_te):
    vect = construir_vectorizador()
    pipe = Pipeline([("tfidf", vect), ("clf", modelo)])
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0

    # latencia de inferencia por mensaje (promedio sobre el test)
    t0 = time.perf_counter()
    ypred = pipe.predict(Xte)
    t_pred_total = time.perf_counter() - t0
    lat_ms = 1000.0 * t_pred_total / len(Xte)

    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    por_idioma = {}
    for lg in ["es", "en", "pt"]:
        m = lang_te == lg
        if m.sum() > 0:
            por_idioma[lg] = {
                "acc": float(accuracy_score(yte[m], ypred[m])),
                "f1": float(f1_score(yte[m], ypred[m], average="macro")),
                "n": int(m.sum()),
            }

    # probabilidades para el analisis hibrido
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(Xte)
        except Exception:
            proba = None

    return {
        "nombre": nombre, "acc": float(acc), "f1_macro": float(f1m),
        "t_fit_s": float(t_fit), "lat_ms": float(lat_ms),
        "por_idioma": por_idioma, "ypred": ypred, "proba": proba,
        "clases": list(pipe.classes_) if hasattr(pipe, "classes_") else None,
        "pipe": pipe,
    }


def matriz_confusion_fig(yte, ypred, titulo, ruta):
    labels = INTENT_ORDER
    cm = confusion_matrix(yte, ypred, labels=labels)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdadera")
    ax.set_title(titulo, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cmn[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def main():
    df = cargar()
    X = df["text"].values
    y = df["intent"].values
    lang = df["lang"].values

    # split estratificado por interseccion intencion x idioma (15% test, como P1)
    strat = np.array([f"{a}|{b}" for a, b in zip(y, lang)])
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.15, random_state=RANDOM_STATE,
                                 stratify=strat)
    Xtr, Xte = X[itr], X[ite]
    ytr, yte = y[itr], y[ite]
    lang_te = lang[ite]
    print(f"Train={len(Xtr)}  Test={len(Xte)}")

    resultados = {"meta": {"n_total": len(X), "n_train": len(Xtr),
                            "n_test": len(Xte), "n_clases": len(INTENT_ORDER)},
                  "modelos": {}, "reglas": {}, "llm": LLM, "hibrido": {}}

    # ---- baseline de reglas ----
    t0 = time.perf_counter()
    ypred_r = np.array([clasificar_reglas(t) for t in Xte])
    lat_r = 1000.0 * (time.perf_counter() - t0) / len(Xte)
    resultados["reglas"] = {
        "acc": float(accuracy_score(yte, ypred_r)),
        "f1_macro": float(f1_score(yte, ypred_r, average="macro")),
        "lat_ms": float(lat_r),
        "por_idioma": {lg: {
            "acc": float(accuracy_score(yte[lang_te == lg], ypred_r[lang_te == lg])),
            "f1": float(f1_score(yte[lang_te == lg], ypred_r[lang_te == lg], average="macro")),
        } for lg in ["es", "en", "pt"]},
    }
    matriz_confusion_fig(yte, ypred_r, "Matriz de confusion - Baseline de reglas",
                          os.path.join(IMG, "cm_reglas.png"))
    print(f"Reglas: acc={resultados['reglas']['acc']:.4f} "
          f"f1={resultados['reglas']['f1_macro']:.4f}")

    # ---- modelos clasicos ----
    evals = {}
    for nombre, modelo in construir_modelos().items():
        r = evaluar_modelo(nombre, modelo, Xtr, ytr, Xte, yte, lang_te)
        evals[nombre] = r
        resultados["modelos"][nombre] = {
            k: r[k] for k in ["acc", "f1_macro", "t_fit_s", "lat_ms", "por_idioma"]
        }
        print(f"{nombre:20s} acc={r['acc']:.4f} f1={r['f1_macro']:.4f} "
              f"lat={r['lat_ms']:.3f}ms fit={r['t_fit_s']:.2f}s")

    # mejor modelo por F1-macro
    mejor = max(evals.values(), key=lambda r: r["f1_macro"])
    resultados["mejor_modelo"] = mejor["nombre"]
    print(f"\nMejor modelo: {mejor['nombre']} (F1={mejor['f1_macro']:.4f})")
    matriz_confusion_fig(
        yte, mejor["ypred"],
        f"Matriz de confusion - {mejor['nombre']} (F1={mejor['f1_macro']:.3f})",
        os.path.join(IMG, "cm_mejor.png"))

    # reporte por clase del mejor modelo
    rep = classification_report(yte, mejor["ypred"], labels=INTENT_ORDER,
                                 output_dict=True, zero_division=0)
    resultados["reporte_mejor"] = {k: rep[k] for k in INTENT_ORDER}

    # ---- figura comparativa de F1-macro ----
    nombres = list(evals.keys())
    f1s = [evals[n]["f1_macro"] for n in nombres]
    orden = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh([nombres[i] for i in orden], [f1s[i] for i in orden], color="#3b6fb0")
    ax.axvline(resultados["reglas"]["f1_macro"], color="#c0392b", ls="--",
               label=f"Reglas ({resultados['reglas']['f1_macro']:.3f})")
    ax.axvline(LLM["f1_macro"], color="#27ae60", ls="--",
               label=f"LLM ({LLM['f1_macro']:.3f})")
    ax.set_xlabel("F1-macro")
    ax.set_xlim(0.0, 1.02)
    ax.set_title("F1-macro por modelo (test, N=%d)" % len(Xte))
    for i, k in enumerate(orden):
        ax.text(f1s[k] + 0.005, i, f"{f1s[k]:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "f1_comparacion.png"), dpi=160)
    plt.close(fig)

    # ---- latencia vs F1 ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in nombres:
        ax.scatter(evals[n]["lat_ms"], evals[n]["f1_macro"], s=60)
        ax.annotate(n, (evals[n]["lat_ms"], evals[n]["f1_macro"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.scatter(LLM["latency_mean_s"] * 1000, LLM["f1_macro"], s=90, marker="*",
               color="#27ae60")
    ax.annotate("LLM", (LLM["latency_mean_s"] * 1000, LLM["f1_macro"]),
                fontsize=8, xytext=(4, -10), textcoords="offset points",
                color="#27ae60")
    ax.set_xscale("log")
    ax.set_xlabel("Latencia por mensaje (ms, escala log)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Compromiso latencia vs calidad")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latencia_f1.png"), dpi=160)
    plt.close(fig)

    # ---- sistema hibrido: mejor clasificador + LLM en baja confianza ----
    # Delega al LLM los mensajes con confianza < umbral. Se estima la exactitud
    # del LLM sobre los delegados con su exactitud empirica global (P1).
    if mejor["proba"] is not None:
        conf = mejor["proba"].max(axis=1)
        clases = np.array(mejor["clases"])
        pred_clf = clases[mejor["proba"].argmax(axis=1)]
        correcto_clf = (pred_clf == yte)
        N = len(yte)
        curva = []
        for tau in np.linspace(0.0, 1.0, 51):
            delega = conf < tau
            cobertura_clf = 1 - delega.mean()
            aciertos = correcto_clf[~delega].sum() + LLM["accuracy"] * delega.sum()
            acc_h = aciertos / N
            costo_rel = delega.mean()   # fraccion que paga costo/latencia LLM
            curva.append({"tau": float(tau),
                           "cobertura_clf": float(cobertura_clf),
                           "acc": float(acc_h),
                           "frac_llm": float(delega.mean()),
                           "costo_rel": float(costo_rel)})
        resultados["hibrido"]["curva"] = curva

        # punto de operacion: menor delegacion que iguala/supera al LLM
        objetivo = LLM["accuracy"]
        op = None
        for p in curva:
            if p["acc"] >= objetivo:
                op = p
        # 'op' = ultimo (mayor tau) que aun cumple; buscamos el de MENOR frac_llm
        candidatos = [p for p in curva if p["acc"] >= objetivo]
        if candidatos:
            op = min(candidatos, key=lambda p: p["frac_llm"])
        resultados["hibrido"]["operacion"] = op
        # ahorro de costo/latencia frente a LLM puro
        if op:
            ahorro = 1 - op["frac_llm"]
            resultados["hibrido"]["ahorro_llamadas_llm"] = float(ahorro)
            print(f"\nHibrido: op tau={op['tau']:.2f} acc={op['acc']:.4f} "
                  f"frac_LLM={op['frac_llm']:.3f} ahorro={ahorro:.1%}")

        # figura de la curva cobertura/exactitud
        taus = [p["tau"] for p in curva]
        accs = [p["acc"] for p in curva]
        fracs = [p["frac_llm"] for p in curva]
        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        ax1.plot(fracs, accs, "-o", ms=3, color="#3b6fb0", label="Exactitud hibrida")
        ax1.axhline(LLM["accuracy"], color="#27ae60", ls="--",
                    label=f"LLM puro ({LLM['accuracy']:.3f})")
        ax1.axhline(mejor["acc"], color="#c0392b", ls=":",
                    label=f"Clasico puro ({mejor['acc']:.3f})")
        ax1.set_xlabel("Fraccion de mensajes delegados al LLM")
        ax1.set_ylabel("Exactitud del sistema")
        ax1.set_title("Compromiso costo-exactitud del sistema hibrido")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG, "hibrido.png"), dpi=160)
        plt.close(fig)

    # ---- distribucion del corpus por idioma/intencion ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    vc_i = df["intent"].value_counts().reindex(INTENT_ORDER)
    axes[0].bar(range(len(vc_i)), vc_i.values, color="#3b6fb0")
    axes[0].set_xticks(range(len(vc_i)))
    axes[0].set_xticklabels(vc_i.index, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Mensajes por intencion")
    vc_l = df["lang"].value_counts()
    axes[1].bar(vc_l.index, vc_l.values, color="#27ae60")
    axes[1].set_title("Mensajes por idioma")
    for i, v in enumerate(vc_l.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "distribucion_corpus.png"), dpi=160)
    plt.close(fig)

    # limpiar objetos no serializables antes de volcar
    with open(os.path.join(HERE, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nresultados.json y figuras escritas en _img/")


if __name__ == "__main__":
    main()
