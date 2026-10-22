import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Previsão de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Modelo de Machine Learning para estimar o consumo de energia (kWh) com base em parâmetros operacionais.")

# =========================================================
# 1. Geração do Dataset Sintético
# =========================================================
@st.cache_data
def gerar_dataset(n_samples=100):
    np.random.seed(42)
    
    velocidade = np.random.uniform(0.5, 5.0, n_samples)          # m/s
    maquinas = np.random.randint(1, 15, n_samples)                # quantidade
    temp = np.random.uniform(18.0, 35.0, n_samples)               # °C
    
    # Fórmula realista + ruído
    consumo = (
        15 * velocidade +
        8 * maquinas +
        1.2 * temp +
        np.random.normal(0, 5, n_samples)
    )
    
    df = pd.DataFrame({
        "Velocidade_Esteira": np.round(velocidade, 2),
        "Maquinas_Ativas": maquinas,
        "Temp_Ambiente": np.round(temp, 1),
        "Consumo_kWh": np.round(consumo, 2)
    })
    return df

df = gerar_dataset(100)

# =========================================================
# 2. Treinamento do Modelo
# =========================================================
@st.cache_resource
def treinar_modelo(df):
    X = df[["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]]
    y = df["Consumo_kWh"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    
    score = modelo.score(X_test, y_test)
    return modelo, score

modelo, r2 = treinar_modelo(df)

# =========================================================
# 3. Sidebar - Controles de Entrada
# =========================================================
st.sidebar.header("⚙️ Parâmetros de Entrada")

velocidade = st.sidebar.number_input(
    "Velocidade da Esteira (m/s)",
    min_value=0.5,
    max_value=5.0,
    value=2.5,
    step=0.1,
    format="%.1f"
)

maquinas = st.sidebar.number_input(
    "Máquinas Ativas",
    min_value=1,
    max_value=15,
    value=6,
    step=1
)

temp = st.sidebar.number_input(
    "Temperatura Ambiente (°C)",
    min_value=18.0,
    max_value=35.0,
    value=25.0,
    step=0.5,
    format="%.1f"
)

# =========================================================
# 4. Previsão
# =========================================================
entrada = pd.DataFrame({
    "Velocidade_Esteira": [velocidade],
    "Maquinas_Ativas": [maquinas],
    "Temp_Ambiente": [temp]
})

consumo_previsto = modelo.predict(entrada)[0]

# =========================================================
# 5. Tela Principal
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Consumo Previsto",
        value=f"{consumo_previsto:.1f} kWh",
        delta=None
    )

with col2:
    st.metric(
        label="R² do Modelo (teste)",
        value=f"{r2:.3f}"
    )

with col3:
    st.metric(
        label="Registros no Dataset",
        value=len(df)
    )

st.markdown("---")

# Gráfico de Importância das Features
st.subheader("📊 Contribuição das Variáveis no Modelo")

importances = modelo.feature_importances_
features = ["Velocidade_Esteira", "Maquinas_Ativas", "Temp_Ambiente"]
cores = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(features, importances, color=cores)
ax.set_xlabel("Importância Relativa")
ax.set_title("Importância das Features (Random Forest)")
ax.set_xlim(0, max(importances) * 1.15)

# Adiciona valores nas barras
for bar, imp in zip(bars, importances):
    ax.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{imp:.3f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()
st.pyplot(fig)

# =========================================================
# Extra: Visualização do Dataset (opcional)
# =========================================================
with st.expander("📂 Ver Dataset de Treinamento"):
    st.dataframe(df, use_container_width=True)
    st.caption("Dataset sintético gerado para fins de demonstração.")

st.markdown("---")
st.caption("Desenvolvido com Streamlit + scikit-learn | Modelo: RandomForestRegressor")
