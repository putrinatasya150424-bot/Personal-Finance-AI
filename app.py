import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key tidak ditemukan! Tambahkan ke .env / Streamlit Secrets.")
    st.stop()

# Init Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.set_page_config(page_title="Personal Finance AI", page_icon="💸", layout="wide")
st.title("💸 Personal Finance AI – Budget Advisor & Spending Insights")
st.write("Upload data pemasukan & pengeluaran kamu, lalu biarkan AI bantu menganalisis keuanganmu! 🔍")

# Model selector
selected_model = st.selectbox(
    "🤖 Pilih Model AI",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    index=0
)

# Upload template: Category | Amount | Type(IN/OUT)
uploaded_file = st.file_uploader("📂 Upload dataset keuangan (.xlsx format)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = ["Category", "Amount", "Type"]
    if not all(col in df.columns for col in required_cols):
        st.error("⚠️ File wajib memiliki kolom: Category, Amount, Type")
        st.stop()

    # Scenario simulation
    df["Optimistic"] = df["Amount"] * np.random.uniform(1.05, 1.15, len(df))
    df["Realistic"] = df["Amount"]
    df["Pessimistic"] = df["Amount"] * np.random.uniform(0.85, 0.95, len(df))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Data Finansial + Skenario")
        st.dataframe(df)

        # Plot: IN vs OUT comparison
        fig = px.bar(
            df,
            x="Category",
            y=["Optimistic", "Realistic", "Pessimistic"],
            color="Type",
            title="📈 Benchmark Pengeluaran & Pemasukan Berdasarkan Skenario",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🤖 Financial Insight AI")

        # Preview for AI
        df_preview = df.head(15).to_string(index=False)

        scenario_prompt = st.text_input(
            "📝 Skenario Keuangan (cth: 'Pengeluaran jajan dikurangi 20%')"
        )

        if st.button("🚀 Analisis Awal AI"):
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """Kamu adalah Personal Finance AI.
                            Bantu mahasiswa mengatur pemasukan & pengeluaran.
                            Berikan analisis budgeting, saving goals, serta tips penghematan."""
                        },
                        {
                            "role": "user",
                            "content": f"Berikut data keuangan:\n{df_preview}\nSkenario: {scenario_prompt}"
                        }
                    ]
                )
                st.success("Analisis berhasil digenerate!")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"⚠️ Gagal request ke AI: {e}")

        # Chat Section
        if "chat" not in st.session_state:
            st.session_state.chat = []

        user_chat = st.text_area("💬 Tanya AI soal keuanganmu:")

        send, reset = st.columns([4,1])

        if send.button("Kirim"):
            if user_chat:
                try:
                    chat_response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": "Kamu ahli personal finance budgeting."},
                            *st.session_state.chat,
                            {"role": "user", "content": f"Data:\n{df_preview}\n\n{user_chat}"}
                        ]
                    )

                    ai_text = chat_response.choices[0].message.content
                    st.session_state.chat.append({"role": "user", "content": user_chat})
                    st.session_state.chat.append({"role": "assistant", "content": ai_text})

                except Exception as e:
                    st.error(f"⚠️ Chat Error: {e}")

        if reset.button("🔄 Reset"):
            st.session_state.chat = []
            st.success("Chat direset!")

        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f"**👤 Kamu:** {msg['content']}")
            else:
                st.markdown(f"**🤖 AI:** {msg['content']}")
