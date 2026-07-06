"""Integra la re-ejecucion real del baseline LLM few-shot (llm_resultados.json,
generada contra la API de OpenRouter sobre el MISMO test set que los modelos
clasicos) y recalcula el sistema hibrido con predicciones reales pareadas
(no un estimado agregado). Actualiza resultados.json y regenera las figuras
afectadas: f1_comparacion, latencia_f1 e hibrido.
"""
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
import experimentos as ex

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "_img")


def main():
    df = ex.cargar()
    X, y, lang = df["text"].values, df["intent"].values, df["lang"].values
    strat = np.array([f"{a}|{b}" for a, b in zip(y, lang)])
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.15, random_state=ex.RANDOM_STATE,
                                 stratify=strat)
    Xtr, Xte = X[itr], X[ite]
    ytr, yte = y[itr], y[ite]
    lang_te = lang[ite]

    with open(os.path.join(HERE, "llm_resultados.json"), encoding="utf-8") as f:
        llm_real = json.load(f)

    # verificar alineacion exacta con el split local
    assert llm_real["verdadero"] == list(yte), "el test set no coincide con llm_resultados.json"
    pred_llm = np.array(llm_real["predicciones"])
    correcto_llm = (pred_llm == yte)
    print("Verificacion de alineacion: OK "
          f"(acc recomputada={correcto_llm.mean():.4f} vs reportada={llm_real['acc']:.4f})")

    # recomputar el mejor modelo clasico (Naive Bayes) con probabilidades
    resultados = json.load(open(os.path.join(HERE, "resultados.json"), encoding="utf-8"))
    mejor_nombre = resultados["mejor_modelo"]
    modelos = ex.construir_modelos()
    r = ex.evaluar_modelo(mejor_nombre, modelos[mejor_nombre], Xtr, ytr, Xte, yte, lang_te)
    conf = r["proba"].max(axis=1)
    clases = np.array(r["clases"])
    pred_clf = clases[r["proba"].argmax(axis=1)]
    correcto_clf = (pred_clf == yte)
    N = len(yte)

    # ---- curva hibrida EXACTA (predicciones reales pareadas, no estimador) ----
    curva = []
    for tau in np.linspace(0.0, 1.0, 51):
        delega = conf < tau
        pred_sistema = np.where(delega, pred_llm, pred_clf)
        acc_h = accuracy_score(yte, pred_sistema)
        f1_h = f1_score(yte, pred_sistema, average="macro")
        curva.append({"tau": float(tau), "frac_llm": float(delega.mean()),
                       "acc": float(acc_h), "f1_macro": float(f1_h)})

    acc_llm_real = llm_real["acc"]
    candidatos = [p for p in curva if p["acc"] >= acc_llm_real]
    op = min(candidatos, key=lambda p: p["frac_llm"]) if candidatos else max(curva, key=lambda p: p["acc"])
    peak = max(curva, key=lambda p: p["acc"])
    print(f"Hibrido (real): tau={op['tau']:.2f} frac_llm={op['frac_llm']:.3f} "
          f"acc={op['acc']:.4f} (iguala LLM real={acc_llm_real:.4f})")
    print(f"Pico hibrido: tau={peak['tau']:.2f} frac_llm={peak['frac_llm']:.3f} acc={peak['acc']:.4f}")

    # ---- actualizar resultados.json ----
    resultados["llm_real_testset"] = {
        "descripcion": "Re-ejecucion real via OpenRouter (gpt-4o-mini, few-shot "
                        "sin RAG) sobre el mismo test set (N=198) usado para los "
                        "modelos clasicos.",
        "acc": llm_real["acc"], "f1_macro": llm_real["f1_macro"],
        "latencia_media_s": llm_real["latencia_media_s"],
        "latencia_min_s": llm_real["latencia_min_s"],
        "latencia_max_s": llm_real["latencia_max_s"],
        "por_idioma": llm_real["por_idioma"],
        "costo_usd_por_1k_mensajes": llm_real["costo_usd_por_1k_mensajes"],
        "n_test": llm_real["n_test"],
    }
    resultados["hibrido_real"] = {"curva": curva, "operacion": op, "pico": peak}
    with open(os.path.join(HERE, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---- figura: F1-macro por modelo (incluye LLM few-shot real) ----
    nombres = list(resultados["modelos"].keys())
    f1s = [resultados["modelos"][n]["f1_macro"] for n in nombres]
    nombres = nombres + ["LLM few-shot (real)"]
    f1s = f1s + [llm_real["f1_macro"]]
    orden = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colores = ["#e67e22" if nombres[i] == "LLM few-shot (real)" else "#3b6fb0" for i in orden]
    ax.barh([nombres[i] for i in orden], [f1s[i] for i in orden], color=colores)
    ax.axvline(resultados["reglas"]["f1_macro"], color="#c0392b", ls="--",
               label=f"Reglas ({resultados['reglas']['f1_macro']:.3f})")
    ax.axvline(resultados["llm"]["f1_macro"], color="#27ae60", ls="--",
               label=f"LLM+RAG, P1 ({resultados['llm']['f1_macro']:.3f})")
    ax.set_xlabel("F1-macro")
    ax.set_xlim(0.7, 1.02)
    ax.set_title("F1-macro por modelo (test, N=198)")
    for i, k in enumerate(orden):
        ax.text(f1s[k] + 0.005, i, f"{f1s[k]:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "f1_comparacion.png"), dpi=160)
    plt.close(fig)

    # ---- figura: latencia vs F1 (incluye LLM few-shot real y LLM+RAG P1) ----
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for n in resultados["modelos"]:
        v = resultados["modelos"][n]
        ax.scatter(v["lat_ms"], v["f1_macro"], s=60, color="#3b6fb0")
        ax.annotate(n, (v["lat_ms"], v["f1_macro"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter(llm_real["latencia_media_s"] * 1000, llm_real["f1_macro"], s=100,
               marker="D", color="#e67e22")
    ax.annotate("LLM few-shot\n(real)", (llm_real["latencia_media_s"] * 1000, llm_real["f1_macro"]),
                fontsize=7, xytext=(6, -18), textcoords="offset points", color="#e67e22")
    ax.scatter(resultados["llm"]["latency_mean_s"] * 1000, resultados["llm"]["f1_macro"],
               s=100, marker="*", color="#27ae60")
    ax.annotate("LLM+RAG (P1)", (resultados["llm"]["latency_mean_s"] * 1000, resultados["llm"]["f1_macro"]),
                fontsize=7, xytext=(4, 6), textcoords="offset points", color="#27ae60")
    ax.set_xscale("log")
    ax.set_xlabel("Latencia por mensaje (ms, escala log)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Compromiso latencia vs calidad")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latencia_f1.png"), dpi=160)
    plt.close(fig)

    # ---- figura: sistema hibrido (curva real, exacta) ----
    fracs = [p["frac_llm"] for p in curva]
    accs = [p["acc"] for p in curva]
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.plot(fracs, accs, "-o", ms=3, color="#3b6fb0", label="Hibrido (real)")
    ax1.axhline(llm_real["acc"], color="#e67e22", ls="--",
                label=f"LLM few-shot puro ({llm_real['acc']:.3f})")
    ax1.axhline(correcto_clf.mean(), color="#c0392b", ls=":",
                label=f"Clasico puro ({correcto_clf.mean():.3f})")
    ax1.scatter([op["frac_llm"]], [op["acc"]], s=110, color="k", zorder=5, label="Operacion")
    ax1.set_xlabel("Fraccion de mensajes delegados al LLM")
    ax1.set_ylabel("Exactitud del sistema")
    ax1.set_title("Compromiso costo-exactitud del sistema hibrido (predicciones reales)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "hibrido.png"), dpi=160)
    plt.close(fig)

    print("\nFiguras y resultados.json actualizados con el baseline LLM real.")


if __name__ == "__main__":
    main()
