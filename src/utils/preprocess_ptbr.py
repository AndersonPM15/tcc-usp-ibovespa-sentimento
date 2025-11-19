"""
Pipeline de preprocessamento avançado para textos em português brasileiro.
Inclui limpeza, normalização, lematização com spaCy, e feature engineering.

Utilizado por notebooks/14_preprocess_ptbr.ipynb

Requer:
    pip install spacy
    python -m spacy download pt_core_news_lg
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import unicodedata
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# spaCy para lematização PT-BR
_SPACY_MODEL = None  # Carregado sob demanda

def _load_spacy_model():
    """Carrega modelo spaCy pt_core_news_lg (lazy loading)."""
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            import spacy
            _SPACY_MODEL = spacy.load('pt_core_news_lg')
            print("✅ Modelo spaCy pt_core_news_lg carregado")
        except OSError:
            print("❌ Modelo spaCy pt_core_news_lg não encontrado!")
            print("   Execute: python -m spacy download pt_core_news_lg")
            raise
    return _SPACY_MODEL


def clean_html(text: str) -> str:
    """Remove tags HTML do texto."""
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def remove_urls(text: str) -> str:
    """Remove URLs do texto."""
    if pd.isna(text):
        return ""
    # Remove http(s) URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove www URLs
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text


def normalize_unicode(text: str) -> str:
    """Normaliza caracteres unicode."""
    if pd.isna(text):
        return ""
    # Normaliza para NFD (decomposição canônica)
    text = unicodedata.normalize('NFD', text)
    # Remove marcas diacríticas exceto as essenciais para português
    # Mantém ã, õ, ç, acentos
    return text


def remove_stopwords_pt(text: str, stopwords: set) -> str:
    """Remove stopwords em português."""
    if pd.isna(text):
        return ""
    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    return ' '.join(filtered)


def clean_text_advanced(text: str) -> str:
    """
    Limpeza avançada de texto:
    - Remove caracteres especiais (mantém acentos PT)
    - Remove números soltos
    - Remove espaços múltiplos
    - Converte para lowercase
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove números soltos (mantém números em contexto: "covid-19", "b3")
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove caracteres especiais (mantém letras PT, espaços e hífens)
    text = re.sub(r'[^a-záàâãéêíóôõúüç\s-]', ' ', text)
    
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def lemmatize_text_spacy(text: str, remove_stopwords: bool = True, remove_punct: bool = True) -> str:
    """
    Aplica lematização em texto PT-BR usando spaCy pt_core_news_lg.
    
    Esta função deve ser usada APÓS a limpeza básica (clean_text_advanced).
    
    Args:
        text: Texto já limpo (lowercase, sem URLs, etc.)
        remove_stopwords: Se True, remove stopwords durante lematização
        remove_punct: Se True, remove pontuação
    
    Returns:
        Texto lematizado
        
    Example:
        >>> text = "os gatos estavam correndo pelos jardins"
        >>> lemmatize_text_spacy(text)
        'gato estar correr jardim'
    """
    if pd.isna(text) or not text.strip():
        return ""
    
    nlp = _load_spacy_model()
    
    # Processar com spaCy
    doc = nlp(text)
    
    # Coletar lemas
    lemmas = []
    for token in doc:
        # Filtros opcionais
        if remove_stopwords and token.is_stop:
            continue
        if remove_punct and token.is_punct:
            continue
        if token.is_space:
            continue
        
        # Adicionar lema (já vem em lowercase do spaCy)
        lemma = token.lemma_.strip()
        if lemma and len(lemma) > 1:  # Ignorar tokens de 1 caractere
            lemmas.append(lemma)
    
    return ' '.join(lemmas)


def preprocess_pipeline(df: pd.DataFrame, 
                       remove_stopwords: bool = True,
                       use_lemmatization: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de preprocessamento PT-BR.
    
    Etapas:
    1. Limpeza HTML
    2. Remoção de URLs
    3. Normalização unicode
    4. Limpeza avançada (lowercase, caracteres especiais)
    5. Remoção de stopwords (opcional)
    6. Lematização com spaCy (opcional, RECOMENDADO)
    7. Tokenização
    
    Args:
        df: DataFrame com coluna 'raw_text'
        remove_stopwords: Se True, remove stopwords PT
        use_lemmatization: Se True, aplica lematização spaCy (PLANO DE PESQUISA)
    
    Returns:
        DataFrame com colunas adicionais: clean_text, lemmatized_text, tokens
    """
    import nltk
    
    # Download stopwords se necessário
    try:
        from nltk.corpus import stopwords
        stop_pt = set(stopwords.words('portuguese'))
    except:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_pt = set(stopwords.words('portuguese'))
    
    print("🧹 Aplicando pipeline de limpeza...")
    
    # 1. Criar coluna de texto completo se não existir
    if 'raw_text' not in df.columns:
        df['raw_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['content'].fillna('')
    
    # 2. Limpeza HTML
    df['text_no_html'] = df['raw_text'].apply(clean_html)
    
    # 3. Remover URLs
    df['text_no_urls'] = df['text_no_html'].apply(remove_urls)
    
    # 4. Normalizar unicode
    df['text_normalized'] = df['text_no_urls'].apply(normalize_unicode)
    
    # 5. Limpeza avançada
    df['clean_text'] = df['text_normalized'].apply(clean_text_advanced)
    
    # 6. Remover stopwords (opcional) - se não usar lematização
    if remove_stopwords and not use_lemmatization:
        df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords_pt(x, stop_pt))
    
    # 7. LEMATIZAÇÃO COM spaCy (NOVO - PLANO DE PESQUISA)
    if use_lemmatization:
        print("🔤 Aplicando lematização com spaCy pt_core_news_lg...")
        print("   (Isso pode demorar alguns minutos...)")
        
        # Lematizar com spaCy (já remove stopwords internamente)
        df['lemmatized_text'] = df['clean_text'].apply(
            lambda x: lemmatize_text_spacy(x, remove_stopwords=remove_stopwords)
        )
        
        # Usar texto lematizado como texto final
        df['clean_text'] = df['lemmatized_text']
        
        print("✅ Lematização concluída")
    
    # 8. Tokenização
    df['tokens'] = df['clean_text'].apply(lambda x: x.split() if x else [])
    df['token_count'] = df['tokens'].apply(len)
    
    # 9. Remover textos muito curtos (< 5 tokens)
    initial_count = len(df)
    df = df[df['token_count'] >= 5].copy()
    removed = initial_count - len(df)
    
    lematizacao_status = "COM lematização spaCy" if use_lemmatization else "SEM lematização"
    print(f"✅ Limpeza concluída ({lematizacao_status})")
    print(f"   Textos muito curtos removidos: {removed:,}")
    print(f"   Registros válidos: {len(df):,}")
    
    # Limpar colunas intermediárias
    if use_lemmatization:
        # Manter lemmatized_text para referência
        df = df.drop(columns=['text_no_html', 'text_no_urls', 'text_normalized'], errors='ignore')
    else:
        df = df.drop(columns=['text_no_html', 'text_no_urls', 'text_normalized'], errors='ignore')
    
    return df


def detect_language(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    Detecta idioma dos textos (sample para performance).
    """
    try:
        from langdetect import detect
        
        print(f"🌐 Detectando idioma (sample de {min(sample_size, len(df))} registros)...")
        
        def safe_detect(text):
            try:
                if pd.isna(text) or len(text) < 20:
                    return 'unknown'
                return detect(text)
            except:
                return 'unknown'
        
        # Detectar em sample
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        sample_df['language'] = sample_df['clean_text'].apply(safe_detect)
        
        # Aplicar ao resto baseado na fonte
        lang_by_source = sample_df.groupby('source')['language'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'pt')
        df['language'] = df['source'].map(lang_by_source).fillna('pt')
        
        lang_counts = df['language'].value_counts()
        print(f"✅ Idiomas detectados:")
        for lang, count in lang_counts.items():
            print(f"   {lang}: {count:,} ({(count/len(df))*100:.1f}%)")
        
        # Filtrar apenas português
        pt_count = len(df)
        df = df[df['language'] == 'pt'].copy()
        removed = pt_count - len(df)
        if removed > 0:
            print(f"⚠️ Removidos {removed:,} textos não-PT")
        
    except ImportError:
        print("⚠️ langdetect não instalado. Assumindo todos os textos em PT.")
        df['language'] = 'pt'
    
    return df


def generate_embeddings(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Gera embeddings usando SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"🧠 Gerando embeddings (batch_size={batch_size})...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = df['clean_text'].fillna('').tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Converter para lista de listas para salvar em parquet
    df['embedding_768'] = embeddings.tolist()
    
    print(f"✅ Embeddings gerados: shape={embeddings.shape}")
    
    return df


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Análise de sentimento usando VADER (PT adaptado) e Transformer PT-BR.
    """
    print("😊 Analisando sentimento...")
    
    # Método 1: VADER (rápido mas menos preciso para PT)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        def get_vader_sentiment(text):
            if pd.isna(text) or not text:
                return 0.0
            scores = analyzer.polarity_scores(text)
            return scores['compound']  # -1 (negativo) a +1 (positivo)
        
        df['sentiment_vader'] = df['clean_text'].apply(get_vader_sentiment)
        print("  ✓ VADER sentiment calculado")
        
    except ImportError:
        print("  ⚠️ vaderSentiment não instalado - pulando VADER")
        df['sentiment_vader'] = 0.0
    
    # Método 2: Sentiment simplificado baseado em keywords financeiras
    positive_keywords = {
        'alta', 'subiu', 'crescimento', 'lucro', 'ganho', 'otimismo', 
        'recuperação', 'expansão', 'avanço', 'recorde', 'positivo'
    }
    negative_keywords = {
        'queda', 'baixa', 'crise', 'prejuízo', 'perda', 'pessimismo',
        'recessão', 'desaceleração', 'risco', 'negativo', 'preocupação'
    }
    
    def get_keyword_sentiment(text):
        if pd.isna(text):
            return 0.0
        words = set(text.lower().split())
        pos_count = len(words & positive_keywords)
        neg_count = len(words & negative_keywords)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    df['sentiment_keywords'] = df['clean_text'].apply(get_keyword_sentiment)
    
    # Sentiment final: média dos dois métodos
    df['sentiment'] = (df['sentiment_vader'] + df['sentiment_keywords']) / 2
    
    print(f"✅ Sentiment calculado:")
    print(f"   Média: {df['sentiment'].mean():.3f}")
    print(f"   Std:   {df['sentiment'].std():.3f}")
    print(f"   Min:   {df['sentiment'].min():.3f}")
    print(f"   Max:   {df['sentiment'].max():.3f}")
    
    return df


def calculate_credibility_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula score de credibilidade baseado em fonte e características do texto.
    """
    print("🔍 Calculando credibility score...")
    
    # Scores por fonte (baseado em reputação)
    source_scores = {
        'valor': 0.9,
        'reuters': 0.95,
        'bloomberg': 0.95,
        'infomoney': 0.85,
        'exame': 0.85,
        'estadao': 0.85,
        'folha': 0.85,
        'g1': 0.75,
        'uol': 0.75,
    }
    
    # Score base por tipo de fonte
    source_type_scores = {
        'rss': 0.8,
        'gdelt': 0.7,
        'gnews': 0.75,
        'newsapi': 0.75
    }
    
    def get_credibility(row):
        # Score por fonte específica
        source_name = row['source'].lower() if 'source' in row else ''
        for key, score in source_scores.items():
            if key in source_name:
                return score
        
        # Score por tipo de fonte
        source_type = row.get('source_type', '').lower()
        base_score = source_type_scores.get(source_type, 0.5)
        
        # Ajustes baseados em características do texto
        token_count = row.get('token_count', 0)
        if token_count > 100:  # Textos mais longos tendem a ser mais completos
            base_score += 0.05
        if token_count < 20:  # Textos muito curtos são suspeitos
            base_score -= 0.1
        
        # Normalizar entre 0 e 1
        return max(0, min(1, base_score))
    
    df['credibility_score'] = df.apply(get_credibility, axis=1)
    
    print(f"✅ Credibility score calculado:")
    print(f"   Média: {df['credibility_score'].mean():.3f}")
    
    return df


def calculate_novelty_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula score de novidade (inversamente proporcional à idade).
    Notícias mais recentes = maior score.
    """
    print("🆕 Calculando novelty score...")
    
    if 'published_at' in df.columns:
        max_date = df['published_at'].max()
        
        # Dias desde publicação
        df['days_old'] = (max_date - df['published_at']).dt.days
        
        # Score exponencial decrescente (half-life = 30 dias)
        df['novelty_score'] = np.exp(-df['days_old'] / 30)
        
        print(f"✅ Novelty score calculado:")
        print(f"   Média: {df['novelty_score'].mean():.3f}")
    else:
        print("⚠️ Coluna 'published_at' não encontrada - novelty = 1.0")
        df['novelty_score'] = 1.0
    
    return df
