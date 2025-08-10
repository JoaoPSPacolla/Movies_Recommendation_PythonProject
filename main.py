import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --- Carrega os dados ---
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dados_filmes_limpo_e_focado.xlsx")
    for feature in ['GENRES', 'KEYWORDS', 'TAGLINE', 'OVERVIEW']:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = (
        df['GENRES'] + ' ' +
        df['KEYWORDS'] + ' ' +
        df['TAGLINE'] + ' ' +
        df['OVERVIEW']
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['TITLE']).drop_duplicates()
    return df, cosine_sim, indices

# --- Função para buscar pôster ---
def get_movie_poster_url(title):
    try:
        api_key = "1457c1b88a022fe3b44f8afc1860d9b9"  # coloque sua chave TMDB aqui
        query = title.replace(" ", "%20")
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                poster_path = data["results"][0]["poster_path"]
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

# --- Função para recomendar filmes ---
def recomendar_filmes(titulo, cosine_sim, df, indices, n=9):
    idx = indices[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['TITLE', 'OVERVIEW']]

# --- Interface Streamlit ---
st.set_page_config(page_title="LigaDs Movie Recommender", layout="centered")
st.title("🎬 Bem-vindo ao recomendador de filmes da LigaDs!")

df, cosine_sim, indices = carregar_dados()

filme_selecionado = st.selectbox(
    "Escolha um filme da lista:",
    options=df['TITLE'].sort_values().unique(),
    index=None,
    placeholder="🔍 Digite o nome do filme..."
)

if st.button("Recomendar filmes") and filme_selecionado:
    st.subheader("Filmes recomendados:")
    recomendados = recomendar_filmes(filme_selecionado, cosine_sim, df, indices)

    for i in range(0, len(recomendados), 3):
        linha = recomendados.iloc[i:i+3]
        cols = st.columns(3)
        for col, (_, row) in zip(cols, linha.iterrows()):
            titulo = row['TITLE']
            descricao = row['OVERVIEW']
            poster_url = get_movie_poster_url(titulo)

            with col:
                if poster_url:
                    descricao_limitada = descricao[:300]
                    card_html = f"""
                    <div style="
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                        border: 1px solid #ccc;
                        border-radius: 10px;
                        padding: 12px;
                        width: 100%;
                        height: 550px;
                        max-width: 380px;
                        margin: 0 auto 20px auto;
                        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
                        background-color: #fdfdfd;
                    ">
                        <img src="{poster_url}" alt="{titulo}" style="width: 100%; height: 350px; object-fit: cover; border-radius: 8px;">
                        <div style="margin-top: 10px;">
                            <h4 style="margin: 0; font-size: 18px; color: #333;">{titulo}</h4>
                            <p style="
                                font-size: 14px;
                                color: #666;
                                text-align: justify;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                display: -webkit-box;
                                -webkit-line-clamp: 4;
                                -webkit-box-orient: vertical;
                                height: 72px;
                            ">
                                {descricao_limitada}
                            </p>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.write(titulo)
                    st.warning("Imagem não encontrada.")
