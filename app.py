import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

R = 8.314462
st.set_page_config(page_title="Cinética y estabilidad", layout="wide")
st.title("Cinética y estabilidad de principios activos")
st.caption("Orden de reacción (método integral), k, t50/t98/t95/t90, Arrhenius y predicción de estabilidad.")


# ---------- funciones de cálculo ----------
def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = stats.linregress(x, y)
    resid = y - (r.slope * x + r.intercept)
    s = np.sqrt((resid ** 2).sum() / (n - 2)) if n > 2 else np.nan
    tc = stats.t.ppf(0.975, n - 2) if n > 2 else np.nan
    return dict(m=r.slope, b=r.intercept, r2=r.rvalue ** 2, n=n, mx=x.mean(),
                sxx=((x - x.mean()) ** 2).sum(), s=s, tc=tc)


def t_for(order, k, A0, q):
    """Tiempo para que quede la fracción q de la concentración inicial."""
    if not k > 0:
        return np.nan
    if order == 0:
        return (1 - q) * A0 / k
    if order == 1:
        return -np.log(q) / k
    return (1 / q - 1) / (A0 * k)


def scatter_fit(x, y, m, b, xl, yl, title):
    xs = np.array([min(x), max(x)])
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="Datos")
    fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Ajuste")
    fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl, height=320,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


# ---------- 1. datos ----------
st.header("1. Datos del ensayo")
TIME = {"segundos": 1, "minutos": 60, "horas": 3600, "días": 86400, "semanas": 604800,
        "meses (30 d)": 2592000, "años (365 d)": 31536000}
CONC = {"mol/L = mol/dm³": ("mol", 1), "mmol/L = mmol/dm³": ("mol", 1e-3), "µmol/L": ("mol", 1e-6),
        "g/L = mg/mL": ("g", 1), "mg/L = µg/mL": ("g", 1e-3), "µg/L": ("g", 1e-6)}
Tt = st.number_input("Temperatura del ensayo (°C)", value=40.0)
sin = st.checkbox("No convertir: uso mis propias unidades (por ejemplo, % p/v)")
if sin:
    x1, x2 = st.columns(2)
    ut_in = ut = x1.text_input("Unidad de tiempo", "semanas")
    uc_in = uc = x2.text_input("Unidad de concentración", "mmol/dm³")
    ft = fc = 1.0
else:
    x1, x2, x3, x4 = st.columns(4)
    ut_in = x1.selectbox("Tiempo: unidad de tus datos", list(TIME), index=4)
    ut = x2.selectbox("Tiempo: unidad de los resultados", list(TIME), index=4)
    uc_in = x3.selectbox("Concentración: unidad de tus datos", list(CONC), index=1)
    uc = x4.selectbox("Concentración: unidad de los resultados", list(CONC), index=1)
    ft = TIME[ut_in] / TIME[ut]
    (ki, fi), (ko, fo) = CONC[uc_in], CONC[uc]
    if ki == ko:
        fc = fi / fo
    else:
        M = st.number_input("Masa molar del fármaco (g/mol), necesaria para pasar entre mol y masa", min_value=0.0, value=0.0)
        if M <= 0:
            st.warning("Ingresa la masa molar para convertir entre unidades molares y de masa.")
            st.stop()
        fc = fi * M / fo if ki == "mol" else fi / (M * fo)
ejemplo = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        "C": [100.05, 82.0, 60.5, 40.06, 22.0, 11.5]})
df = st.data_editor(ejemplo, num_rows="dynamic", width="stretch", key="datos",
                    column_config={"t": st.column_config.NumberColumn(f"Tiempo ({ut_in})", format="%.4g", step=0.01),
                                   "C": st.column_config.NumberColumn(f"Concentración ({uc_in})", format="%.6g", step=0.0001)})
df = df.astype(float)
st.caption("Los datos de ejemplo son el ejercicio del piroxicam (semanas, mmol/dm³). Bórralos y escribe los tuyos. "
           f"Los cálculos se hacen en {ut} y {uc}.")
df = df.dropna()

if len(df) < 3:
    st.warning("Ingresa al menos 3 parejas tiempo–concentración.")
    st.stop()

t = df["t"].to_numpy(float) * ft
c = df["C"].to_numpy(float) * fc
positivo = (c > 0).all()

# ---------- 2. orden de reacción ----------
st.header("2. Orden de reacción y parámetros cinéticos")
modelos = [dict(o=0, n="Orden cero", yl="[A]", y=c, ku=f"{uc}/{ut}", ok=True),
           dict(o=1, n="Orden uno", yl="ln[A]", y=np.log(c) if positivo else None, ku=f"1/{ut}", ok=positivo),
           dict(o=2, n="Orden dos", yl="1/[A]", y=1 / c if positivo else None, ku=f"1/({uc}·{ut})", ok=positivo)]
for m in modelos:
    if m["ok"]:
        m["f"] = fit(t, m["y"])
        m["k"] = m["f"]["m"] if m["o"] == 2 else -m["f"]["m"]
        b = m["f"]["b"]
        m["A0"] = b if m["o"] == 0 else np.exp(b) if m["o"] == 1 else 1 / b
validos = sorted([m for m in modelos if m["ok"]], key=lambda m: -m["f"]["r2"])
mejor = validos[0]
ambiguo = len(validos) > 1 and mejor["f"]["r2"] - validos[1]["f"]["r2"] < 0.005

msg = f"**Modelo más probable: {mejor['n']}** (R² = {mejor['f']['r2']:.4f})."
if ambiguo:
    deg = (1 - c[np.argmax(t)] / c[np.argmin(t)]) * 100
    st.warning(msg + f" La diferencia de R² con el segundo modelo es menor a 0,005: los datos no distinguen bien el orden. "
               f"En este ensayo se degradó solo {deg:.1f} % del principio activo; con poca degradación las tres "
               "funciones se parecen a una recta (diap. 89-90). Extiende el ensayo hasta mayor conversión, o "
               "confirma el orden con el método de la semivida (diap. 91) usando varias concentraciones iniciales.")
else:
    st.success(msg)

cols = st.columns(3)
for col, m in zip(cols, modelos):
    if m["ok"]:
        col.plotly_chart(scatter_fit(t, m["y"], m["f"]["m"], m["f"]["b"], f"t ({ut})", m["yl"],
                                     f"{m['yl']} vs t"), width="stretch")
    else:
        col.info("No se puede calcular con concentraciones ≤ 0.")

tabla = pd.DataFrame([{"Modelo": m["n"], "Pendiente": m["f"]["m"], "Intercepto": m["f"]["b"],
                       "R²": m["f"]["r2"], "k": m["k"], "Unidades de k": m["ku"]}
                      for m in validos])
st.dataframe(tabla, width="stretch", hide_index=True)

o, k, A0 = mejor["o"], mejor["k"], mejor["A0"]
params = pd.DataFrame({"Parámetro": ["k", "[A]0 ajustada", "t50", "t98", "t95", "t90"],
                       "Valor": [f"{k:.4g} {mejor['ku']}", f"{A0:.4g} {uc}"] +
                                [f"{t_for(o, k, A0, q):.4g} {ut}" for q in (.5, .98, .95, .9)]})
st.table(params)

# ---------- 3. Arrhenius ----------
st.header("3. Módulo Arrhenius")
st.caption(f"Importante: las k de esta tabla deben estar en 1/{ut}, la misma unidad de tiempo del paso 1.")
if "arr" not in st.session_state:
    st.session_state.arr = pd.DataFrame({"T_C": pd.Series(dtype=float), "k": pd.Series(dtype=float)})
b1, b2 = st.columns(2)
if b1.button("Agregar k del análisis actual"):
    a = st.session_state.arr
    a = a[a["T_C"] != Tt]
    st.session_state.arr = pd.concat([a, pd.DataFrame({"T_C": [Tt], "k": [k]})]).sort_values("T_C").reset_index(drop=True)
if b2.button("Cargar ejemplo de Arrhenius (k en 1/h)"):
    st.session_state.arr = pd.DataFrame({"T_C": [30.0, 40.0, 50.0, 60.0], "k": [0.0183, 0.0498, 0.131, 0.309]})
arr = st.data_editor(st.session_state.arr, num_rows="dynamic", width="stretch", key="arr_ed").dropna()
arr = arr[arr["k"] > 0]

F = FE = None
if len(arr) >= 2:
    invT = 1 / (arr["T_C"].to_numpy(float) + 273.15)
    lnk = np.log(arr["k"].to_numpy(float))
    F = fit(invT, lnk)
    Ea = -F["m"] * R / 1000
    ic = F["tc"] * F["s"] / np.sqrt(F["sxx"]) * R / 1000 if F["n"] > 2 else np.nan
    st.dataframe(pd.DataFrame({"T (°C)": arr["T_C"], "1/T (K⁻¹)": invT, "k": arr["k"], "ln k": lnk}),
                 width="stretch", hide_index=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Ea (kJ/mol)", f"{Ea:.4g}", None if np.isnan(ic) else f"± {ic:.3g} (IC 95 %)", delta_color="off")
    a2.metric("A (factor preexponencial)", f"{np.exp(F['b']):.4g}")
    a3.metric("R²", f"{F['r2']:.4f}")
    st.plotly_chart(scatter_fit(1000 * invT, lnk, F["m"] / 1000, F["b"], "1000/T (K⁻¹)", "ln k",
                                "Gráfico de Arrhenius"), width="stretch")
    # Teoría del estado de transición (diap. 80-82): ln(k/T) = ln(kB/h) + ΔS‡/R - ΔH‡/(R·T)
    KB, HP = 1.380649e-23, 6.62607015e-34
    Tk = arr["T_C"].to_numpy(float) + 273.15
    FE = fit(invT, lnk - np.log(Tk))
    DH = -FE["m"] * R / 1000
    st.subheader("Estado de transición (Eyring)")
    e1, e2, e3 = st.columns(3)
    e1.metric("ΔH‡ (kJ/mol)", f"{DH:.4g}")
    if ut in TIME and o == 1:
        DS = R * (FE["b"] - np.log(TIME[ut]) - np.log(KB / HP))
        e2.metric("ΔS‡ (J/mol·K)", f"{DS:.4g}")
        e3.metric(f"ΔG‡ a {Tt:g} °C (kJ/mol)", f"{DH - (Tt + 273.15) * DS / 1000:.4g}")
    else:
        e2.metric("ΔS‡ (J/mol·K)", "—")
        st.caption("ΔS‡ y ΔG‡ se calculan solo para orden uno con tiempo en unidades conocidas (k debe pasarse a s⁻¹).")
    st.caption(f"R² del gráfico de Eyring = {FE['r2']:.4f}. La diapositiva 82 asume ΔH‡ ≈ Ea (aquí Ea = {Ea:.4g} kJ/mol).")
else:
    st.info("Ingresa al menos 2 temperaturas con su k.")

# ---------- 4. predicción ----------
st.header("4. Predicción de estabilidad")
if F is None:
    st.stop()
modelo = st.radio("Modelo para predecir k(T)", ["Arrhenius", "Eyring (estado de transición)"], horizontal=True)
eyr = modelo.startswith("Eyring")
FX = FE if eyr else F


def k_at(T):
    x = 1 / (T + 273.15)
    l = FX["b"] + FX["m"] * x
    w = FX["tc"] * FX["s"] * np.sqrt(1 / FX["n"] + (x - FX["mx"]) ** 2 / FX["sxx"]) if FX["n"] > 2 else 0
    sc = (T + 273.15) if eyr else 1.0
    return sc * np.exp(l), sc * np.exp(l - w), sc * np.exp(l + w)


Tmin, Tmax = arr["T_C"].min(), arr["T_C"].max()
if Tmax - Tmin > 100:
    st.warning("El intervalo de temperaturas ensayado supera 100 K: A solo puede considerarse constante en ≤ 50–100 K (diap. 74).")
T0 = st.slider("Temperatura de almacenamiento (°C)", 0, 60, 25)
if not Tmin <= T0 <= Tmax:
    st.warning(f"{T0} °C está fuera del intervalo ensayado ({Tmin:g}–{Tmax:g} °C): es una extrapolación y asume el mismo mecanismo.")
K0 = k_at(T0)
razon = k / k_at(Tt)[0]
if not 0.67 < razon < 1.5:
    st.warning(f"Revisa las unidades: según el modelo, k a {Tt:g} °C sería {k_at(Tt)[0]:.4g}, pero el análisis del paso 2 dio "
               f"{k:.4g} {mejor['ku']}. Las k de la tabla de Arrhenius deben estar en la misma unidad de tiempo que el paso 1.")
st.metric(f"t90 a {T0} °C ({ut})", f"{t_for(o, K0[0], A0, .9):.4g}")

a_meses = TIME[ut] / TIME["meses (30 d)"] if ut in TIME else None
# condiciones de los estudios de estabilidad (diap. 60): (nombre, T, duración del estudio en meses)
esc = [("Prolongada 25 ± 2 °C / 60 ± 5 % HR (12 meses)", 25, 12), ("Alternativa 30 ± 2 °C / 60 ± 5 % HR", 30, None),
       ("Acelerada 40 ± 2 °C / 75 ± 5 % HR (6 meses)", 40, 6), ("Refrigerador", 5, None), ("Temperatura elegida", T0, None)]
filas = []
for nombre, T, dur in esc:
    kk, lo, hi = k_at(T)
    t90c, t90_cons = t_for(o, kk, A0, .9), t_for(o, hi, A0, .9)
    r90 = f"{t90c:.4g}" + (f" ({t_for(o, hi, A0, .9):.4g}–{t_for(o, lo, A0, .9):.4g})" if FX["n"] > 2 else "")
    fila = {"Escenario": nombre, "T (°C)": T, f"k ({mejor['ku']})": f"{kk:.4g}",
            f"t50 ({ut})": f"{t_for(o, kk, A0, .5):.4g}", "t98": f"{t_for(o, kk, A0, .98):.4g}",
            "t95": f"{t_for(o, kk, A0, .95):.4g}", "t90 (IC 95 %)": r90,
            "¿Extrapolado?": "Sí" if not Tmin <= T <= Tmax else "No"}
    if a_meses:
        fila["t90 (meses)"] = f"{t90c * a_meses:.3g}"
        fila["¿Cubre el estudio?"] = ("Sí" if t90_cons * a_meses >= dur else "No") if dur else "—"
    filas.append(fila)
st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
st.caption("«Cubre el estudio» compara el límite conservador del t90 (IC 95 %) con la duración del ensayo de estabilidad.")

sens = "muy sensible a la temperatura" if Ea > 80 else "moderadamente sensible a la temperatura" if Ea > 40 else "poco sensible a la temperatura"
t25 = t_for(o, k_at(25)[0], A0, .9)
texto = (f"El mejor ajuste cinético es de {mejor['n'].lower()} (R² = {mejor['f']['r2']:.4f}), con k = {k:.4g} "
         f"{mejor['ku']} a {Tt:g} °C. La energía de activación es {Ea:.4g} kJ/mol (ΔH‡ = {DH:.4g} kJ/mol), por lo que la "
         f"degradación es {sens}. Con {modelo.split(' ')[0]}, a 25 °C t90 ≈ {t25:.4g} {ut}")
if a_meses:
    texto += f" ({t25 * a_meses:.3g} meses), " + ("suficiente" if t25 * a_meses >= 12 else "insuficiente") + " para cubrir los 12 meses del estudio prolongado"
texto += ". Las extrapolaciones asumen el mismo mecanismo de degradación en todo el intervalo, y el mismo pH, luz, humedad y oxígeno del ensayo."
if ambiguo:
    texto += " Atención: el orden de reacción es ambiguo y eso afecta todos los tiempos."
if F["n"] < 3:
    texto += " Con solo 2 temperaturas no hay intervalo de confianza."
st.info("**Conclusión automática.** " + texto)
