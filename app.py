import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Sales Intelligence", layout="wide", page_icon="🛒")

# ── THEME ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

* { box-sizing: border-box; }

.stApp { background: #070b14; color: #dde4f0; font-family: 'Outfit', sans-serif; }

h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.03em; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1525 0%, #111a2e 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1rem 1.4rem;
}

[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1a2540;
}

.section-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3b82f6;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.insight-card {
    background: linear-gradient(135deg, #0d1525, #111e38);
    border: 1px solid #1e3058;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #94b8ff;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0a0f1e; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #7090b0; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.stTabs [aria-selected="true"] { background: #1a3060 !important; color: #60a0ff !important; }

div[data-testid="stDataFrame"] { border: 1px solid #1a2540; border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/amazon.csv")
    df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
    df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].replace(',', '', regex=True), errors='coerce')
    df.dropna(subset=['discounted_price', 'rating', 'rating_count'], inplace=True)

    df['demand_score'] = df['rating'] * np.log1p(df['rating_count'])
    df['value_score'] = df['rating'] / (df['discounted_price'] + 1) * 1000
    df['savings'] = df['actual_price'] - df['discounted_price']
    df['main_category'] = df['category'].apply(lambda x: str(x).split('|')[0].strip())
    df['sub_category'] = df['category'].apply(lambda x: str(x).split('|')[1].strip() if '|' in str(x) else '')
    df['rating_band'] = pd.cut(df['rating'], bins=[0,2,3,4,4.5,5],
                                labels=['<2★','2–3★','3–4★','4–4.5★','4.5–5★'])
    df['price_tier'] = pd.qcut(df['discounted_price'], q=4,
                                labels=['Budget','Mid','Premium','Luxury'])
    return df

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 AI Sales Intel")
    st.markdown("---")
    categories = sorted(df['main_category'].unique())
    selected_cats = st.multiselect("Categories", categories, default=categories[:4])
    st.markdown("---")
    price_range = st.slider("Price Range (₹)", int(df['discounted_price'].min()),
                             int(df['discounted_price'].max()),
                             (0, int(df['discounted_price'].quantile(0.9))))
    min_rating = st.slider("Min Rating", 1.0, 5.0, 3.5, 0.1)
    st.markdown("---")
    sort_by = st.selectbox("Sort By", ["demand_score", "value_score", "discounted_price", "rating"])
    top_n = st.slider("Top N Products", 5, 30, 10)

# ── FILTER ────────────────────────────────────────────────────────────────────
fdf = df[
    df['main_category'].isin(selected_cats) &
    df['discounted_price'].between(*price_range) &
    (df['rating'] >= min_rating)
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-tag">// Amazon Product Intelligence</p>', unsafe_allow_html=True)
st.title("AI Sales Dashboard")

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Products", f"{len(fdf):,}", f"{len(fdf)-len(df)//2:+,}")
k2.metric("Avg Rating", f"{fdf['rating'].mean():.2f} ★")
k3.metric("Avg Price", f"₹{int(fdf['discounted_price'].mean()):,}")
k4.metric("Avg Discount", f"{fdf['discount_percentage'].mean():.1f}%" if 'discount_percentage' in fdf else "N/A")
k5.metric("Avg Savings", f"₹{int(fdf['savings'].mean()):,}" if 'savings' in fdf else "N/A")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏆 Top Products", "📈 Analytics", "🤖 AI Predictor", "📋 Data"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-tag">// Category × Rating Heatmap</p>', unsafe_allow_html=True)
        heat = fdf.groupby(['main_category','rating_band']).size().unstack(fill_value=0)
        fig = px.imshow(heat, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Rating Band", y="Category", color="Count"))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-tag">// Price Tier Distribution</p>', unsafe_allow_html=True)
        tier_counts = fdf['price_tier'].value_counts().reset_index()
        fig = px.pie(tier_counts, names='price_tier', values='count',
                     color_discrete_sequence=['#1e40af','#2563eb','#3b82f6','#60a5fa'], hole=0.55)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#aac0e0',
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-tag">// Products per Category</p>', unsafe_allow_html=True)
        cat_counts = fdf['main_category'].value_counts().head(10).reset_index()
        fig = px.bar(cat_counts, x='count', y='main_category', orientation='h',
                     color='count', color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<p class="section-tag">// Rating Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x='rating', nbins=20, color_discrete_sequence=['#3b82f6'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TOP PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    top_df = fdf.nlargest(top_n, sort_by)

    st.markdown(f'<p class="section-tag">// Top {top_n} Products by {sort_by}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(top_df, x=sort_by, y='product_name', orientation='h',
                     color='rating', color_continuous_scale='Blues',
                     hover_data=['discounted_price', 'rating_count'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', yaxis={'categoryorder': 'total ascending'},
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Quick Insights**")
        for _, row in top_df.head(5).iterrows():
            st.markdown(f"""<div class="insight-card">
                <b>{row['product_name'][:40]}...</b><br>
                ⭐ {row['rating']} &nbsp;|&nbsp; ₹{int(row['discounted_price']):,}
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-tag">// Price vs Rating (Top Products)</p>', unsafe_allow_html=True)
    fig = px.scatter(top_df, x='rating', y='discounted_price', size='rating_count',
                     color='demand_score', hover_name='product_name',
                     color_continuous_scale='Blues', size_max=40)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-tag">// Avg Price by Category</p>', unsafe_allow_html=True)
        avg_price = fdf.groupby('main_category')['discounted_price'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_price, x='main_category', y='discounted_price', color='discounted_price',
                     color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', xaxis_tickangle=-30, margin=dict(l=0,r=0,t=10,b=60))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-tag">// Discount % vs Savings</p>', unsafe_allow_html=True)
        if 'discount_percentage' in fdf and 'savings' in fdf:
            fig = px.scatter(fdf.sample(min(500, len(fdf))), x='discount_percentage', y='savings',
                             color='rating', size='rating_count', size_max=20,
                             color_continuous_scale='Blues', hover_name='product_name')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-tag">// Demand Score by Category (Box Plot)</p>', unsafe_allow_html=True)
    fig = px.box(fdf, x='main_category', y='demand_score', color='main_category',
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#aac0e0', showlegend=False, xaxis_tickangle=-30,
                      margin=dict(l=0,r=0,t=10,b=60))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-tag">// Correlation Matrix</p>', unsafe_allow_html=True)
    num_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'demand_score', 'value_score']
    corr_cols = [c for c in num_cols if c in fdf.columns]
    corr = fdf[corr_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(fdf) < 30:
        st.warning("Not enough data to train a model. Adjust filters.")
    else:
        le = LabelEncoder()
        fdf['cat_enc'] = le.fit_transform(fdf['main_category'])

        FEATURES = ['rating', 'rating_count', 'discount_percentage', 'cat_enc'] if 'discount_percentage' in fdf else ['rating', 'rating_count', 'cat_enc']
        FEATURES = [f for f in FEATURES if f in fdf.columns]
        X = fdf[FEATURES]
        y = fdf['discounted_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown('<p class="section-tag">// Random Forest</p>', unsafe_allow_html=True)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_r2 = r2_score(y_test, rf_preds)
            rf_mae = mean_absolute_error(y_test, rf_preds)
            st.metric("R² Score", f"{rf_r2:.3f}")
            st.metric("MAE", f"₹{int(rf_mae):,}")

        with col_m2:
            st.markdown('<p class="section-tag">// Gradient Boosting</p>', unsafe_allow_html=True)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)
            gb_preds = gb.predict(X_test)
            gb_r2 = r2_score(y_test, gb_preds)
            gb_mae = mean_absolute_error(y_test, gb_preds)
            st.metric("R² Score", f"{gb_r2:.3f}")
            st.metric("MAE", f"₹{int(gb_mae):,}")

        best_model = rf if rf_r2 >= gb_r2 else gb
        best_name = "Random Forest" if rf_r2 >= gb_r2 else "Gradient Boosting"
        st.info(f"✅ Best model: **{best_name}** (R² = {max(rf_r2, gb_r2):.3f})")

        st.markdown("---")
        st.markdown('<p class="section-tag">// Feature Importance</p>', unsafe_allow_html=True)
        importance_df = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_}).sort_values('importance', ascending=True)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h', color='importance',
                     color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-tag">// Predict New Product Price</p>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            p_rating = st.slider("Rating", 1.0, 5.0, 4.2, 0.1)
        with p2:
            p_reviews = st.slider("Reviews", 1, 50000, 1000, 100)
        with p3:
            p_discount = st.slider("Discount %", 0, 90, 20) if 'discount_percentage' in FEATURES else 20
        with p4:
            p_cat = st.selectbox("Category", sorted(fdf['main_category'].unique()))

        p_cat_enc = le.transform([p_cat])[0]
        input_data = {'rating': p_rating, 'rating_count': p_reviews, 'cat_enc': p_cat_enc}
        if 'discount_percentage' in FEATURES:
            input_data['discount_percentage'] = p_discount
        input_row = [[input_data[f] for f in FEATURES]]
        pred = best_model.predict(input_row)[0]

        col_r1, col_r2 = st.columns(2)
        col_r1.success(f"💰 Predicted Price: ₹{int(pred):,}")
        demand_est = p_rating * np.log1p(p_reviews)
        col_r2.info(f"📈 Estimated Demand Score: {demand_est:.1f}")

        st.markdown('<p class="section-tag">// Actual vs Predicted</p>', unsafe_allow_html=True)
        avp = pd.DataFrame({'Actual': y_test.values, 'Predicted': best_model.predict(X_test)}).head(100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avp['Actual'], y=avp['Predicted'], mode='markers',
                                  marker=dict(color='#3b82f6', size=6, opacity=0.7)))
        fig.add_trace(go.Scatter(x=[avp['Actual'].min(), avp['Actual'].max()],
                                  y=[avp['Actual'].min(), avp['Actual'].max()],
                                  mode='lines', line=dict(color='#ef4444', dash='dash'), name='Ideal'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#aac0e0', xaxis_title='Actual', yaxis_title='Predicted',
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-tag">// Raw Data Explorer</p>', unsafe_allow_html=True)

    search = st.text_input("🔍 Search product name")
    show_cols = st.multiselect("Columns", fdf.columns.tolist(),
                                default=['product_name','main_category','discounted_price','actual_price','rating','rating_count','demand_score'])

    view_df = fdf[show_cols].copy()
    if search:
        view_df = view_df[view_df['product_name'].str.contains(search, case=False, na=False)] if 'product_name' in show_cols else view_df

    st.dataframe(view_df.head(50), use_container_width=True)
    csv = view_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "filtered_products.csv", "text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p style="text-align:center;font-family:Space Mono,monospace;font-size:0.7rem;color:#334;letter-spacing:0.1em;">AI SALES INTELLIGENCE // AMAZON PRODUCT ANALYTICS // ML POWERED</p>', unsafe_allow_html=True)