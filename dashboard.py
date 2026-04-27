"""
═══════════════════════════════════════════════════════════
  MANIPÜLASYON HARİTASI — STREAMLIT DASHBOARD
  CSV Veri Analizi
═══════════════════════════════════════════════════════════

Çalıştırmak için:  streamlit run dashboard.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Sayfa ayarları ──
st.set_page_config(
    page_title="Manipülasyon Tespit Sistemi",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 12px; border: 1px solid #0f3460; }
    .block-container { padding-top: 1rem; }
    h1 { color: #e94560; }
    h2 { color: #0f3460; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  VERİ YÜKLEME (cached)
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """CSV dosyasını yükle ve işle"""
    csv_path = os.path.join(os.path.dirname(__file__), 'manipulative_results.csv')
    
    if not os.path.exists(csv_path):
        st.error("⚠ manipulative_results.csv bulunamadı!")
        st.stop()
    
    df = pd.read_csv(csv_path)
    
    # Label sütunu ekle (manipulation_score'a göre sınıflandırma)
    df['label'] = df['manipulation_score'].apply(lambda x: 
        'Manipülatif' if x >= 0.7 else ('Şüpheli' if x >= 0.5 else 'Organik')
    )
    
    return df


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════

def render_sidebar():
    """Sidebar filtrelerini oluştur"""
    st.sidebar.title("🔍 Filtreler")

    df = load_data()

    # Dil filtresi
    languages = ['Tümü'] + sorted(df['language'].dropna().unique().tolist())
    sel_lang = st.sidebar.selectbox("Dil", languages)

    # Tema filtresi
    themes = ['Tümü'] + sorted(df['primary_theme'].dropna().unique().tolist())
    sel_theme = st.sidebar.selectbox("Tema", themes)

    # Label filtresi
    labels = ['Tümü', 'Manipülatif', 'Şüpheli', 'Organik']
    sel_label = st.sidebar.selectbox("Sınıf", labels)

    # Score aralığı
    score_range = st.sidebar.slider("Manipülasyon Skoru Aralığı", 0.0, 1.0, (0.0, 1.0), 0.01)

    # Filtrele
    mask = pd.Series(True, index=df.index)
    if sel_lang != 'Tümü':
        mask &= df['language'] == sel_lang
    if sel_theme != 'Tümü':
        mask &= df['primary_theme'] == sel_theme
    if sel_label != 'Tümü':
        mask &= df['label'] == sel_label
    mask &= df['manipulation_score'].between(score_range[0], score_range[1])

    return df[mask], df


# ═══════════════════════════════════════════════════════════
#  ANA SAYFA
# ═══════════════════════════════════════════════════════════

def main():
    filtered_df, full_df = render_sidebar()

    # ── Başlık ──
    st.title("🔍 Sosyal Medya Manipülasyon Tespit Sistemi")
    st.markdown("*CSV Veri Analizi — Unsupervised Anomaly Detection Pipeline*")

    # ── Üst metrikler ──
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(full_df)
    manip = len(full_df[full_df['label'] == 'Manipülatif'])
    susp = len(full_df[full_df['label'] == 'Şüpheli'])
    org = len(full_df[full_df['label'] == 'Organik'])
    avg_score = full_df['manipulation_score'].mean()

    col1.metric("📊 Toplam Post", f"{total:,}")
    col2.metric("🔴 Manipülatif", f"{manip:,}", f"{manip/max(total,1)*100:.1f}%")
    col3.metric("🟡 Şüpheli", f"{susp:,}", f"{susp/max(total,1)*100:.1f}%")
    col4.metric("🟢 Organik", f"{org:,}", f"{org/max(total,1)*100:.1f}%")
    col5.metric("📈 Ort. Skor", f"{avg_score:.3f}")

    st.markdown("---")

    # ═══ TABS ═══
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Genel Bakış", "🌍 Dil & Tema", "📈 İstatistikler",
        "📋 Detaylı Veri", "🧪 Canlı Tahmin"
    ])

    # ── TAB 1: Genel Bakış ──
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Manipülasyon Skoru Dağılımı")
            fig = px.histogram(
                filtered_df, x='manipulation_score', nbins=50,
                color='label',
                color_discrete_map={'Manipülatif': '#e94560', 'Şüpheli': '#f5a623', 'Organik': '#44bd32'},
                template='plotly_dark',
            )
            fig.update_layout(height=400, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Sınıf Dağılımı")
            label_counts = filtered_df['label'].value_counts().reset_index()
            label_counts.columns = ['Sınıf', 'Adet']
            fig = px.pie(
                label_counts, values='Adet', names='Sınıf',
                color='Sınıf',
                color_discrete_map={'Manipülatif': '#e94560', 'Şüpheli': '#f5a623', 'Organik': '#44bd32'},
                template='plotly_dark', hole=0.4,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Tema dağılımı
        st.subheader("Tema Bazlı Manipülasyon Oranı")
        theme_stats = filtered_df.groupby('primary_theme').agg(
            avg_score=('manipulation_score', 'mean'),
            count=('manipulation_score', 'count'),
        ).reset_index()
        theme_stats = theme_stats[theme_stats['count'] > 0].sort_values('avg_score', ascending=True)
        
        if len(theme_stats) > 0:
            fig = px.bar(
                theme_stats.tail(20), x='avg_score', y='primary_theme', orientation='h',
                color='avg_score', color_continuous_scale='RdYlGn_r',
                template='plotly_dark', labels={'avg_score': 'Ort. Skor', 'primary_theme': 'Tema'},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Dil & Tema ──
    with tab2:
        st.subheader("Dil × Tema Manipülasyon Haritası")

        heatmap_data = filtered_df.groupby(['language', 'primary_theme']).agg(
            avg_score=('manipulation_score', 'mean'),
            count=('manipulation_score', 'count'),
        ).reset_index()
        
        # En çok post olan kombinasyonlar
        top = heatmap_data.nlargest(100, 'count')
        if len(top) > 0:
            pivot = top.pivot_table(index='language', columns='primary_theme', values='avg_score', fill_value=0)

            fig = px.imshow(
                pivot, color_continuous_scale='RdYlGn_r',
                template='plotly_dark', aspect='auto',
                labels={'color': 'Ort. Manipülasyon Skoru'},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Dil bazlı metrikler
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Dil Bazlı Dağılım")
            lang_stats = filtered_df.groupby('language').agg(
                count=('manipulation_score', 'count'),
                avg_score=('manipulation_score', 'mean'),
            ).reset_index().sort_values('count', ascending=False).head(15)

            fig = px.bar(lang_stats, x='language', y='count', color='avg_score',
                        color_continuous_scale='RdYlGn_r', template='plotly_dark')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Tema Bazlı Dağılım")
            theme_count_stats = filtered_df.groupby('primary_theme').agg(
                count=('manipulation_score', 'count'),
                avg_score=('manipulation_score', 'mean'),
            ).reset_index().sort_values('count', ascending=False).head(15)

            fig = px.bar(theme_count_stats, x='primary_theme', y='count', color='avg_score',
                        color_continuous_scale='RdYlGn_r', template='plotly_dark')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: İstatistikler ──
    with tab3:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Skor Dağılımı İstatistikleri")
            stats_data = {
                'İstatistik': ['Minimum', 'Q1 (25%)', 'Medyan', 'Ortalama', 'Q3 (75%)', 'Maksimum', 'Std. Sapma'],
                'Değer': [
                    f"{filtered_df['manipulation_score'].min():.4f}",
                    f"{filtered_df['manipulation_score'].quantile(0.25):.4f}",
                    f"{filtered_df['manipulation_score'].median():.4f}",
                    f"{filtered_df['manipulation_score'].mean():.4f}",
                    f"{filtered_df['manipulation_score'].quantile(0.75):.4f}",
                    f"{filtered_df['manipulation_score'].max():.4f}",
                    f"{filtered_df['manipulation_score'].std():.4f}",
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        with col_b:
            st.subheader("Sınıf Dağılım Yüzdeleri")
            class_dist = filtered_df['label'].value_counts()
            class_pct = {
                'Sınıf': class_dist.index.tolist(),
                'Yüzde (%)': (class_dist.values / class_dist.sum() * 100).round(2).tolist()
            }
            st.dataframe(pd.DataFrame(class_pct), use_container_width=True)

        st.subheader("Box Plot — Tema Bazlı Skor Dağılımı")
        fig = px.box(
            filtered_df, x='primary_theme', y='manipulation_score',
            color='label',
            template='plotly_dark',
            labels={'primary_theme': 'Tema', 'manipulation_score': 'Manipülasyon Skoru'},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Box Plot — Dil Bazlı Skor Dağılımı")
        fig = px.box(
            filtered_df, x='language', y='manipulation_score',
            color='label',
            template='plotly_dark',
            labels={'language': 'Dil', 'manipulation_score': 'Manipülasyon Skoru'},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Detaylı Veri ──
    with tab4:
        st.subheader("📋 Filtrelenmiş Veri Tablosu")
        
        # Sıralama seçeneği
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sırala:", ['manipulation_score', 'original_text', 'language', 'primary_theme'])
        with col2:
            sort_order = st.selectbox("Sıra:", ['Azalan', 'Artan'])
        with col3:
            show_limit = st.number_input("Göster:", min_value=10, max_value=500, value=100, step=10)
        
        ascending = sort_order == 'Artan'
        display_df = filtered_df.sort_values(by=sort_by, ascending=ascending).head(show_limit).copy()
        display_df['original_text'] = display_df['original_text'].str[:100]
        display_df['manipulation_score'] = display_df['manipulation_score'].round(4)
        
        st.dataframe(
            display_df[['original_text', 'language', 'primary_theme', 'manipulation_score', 'label']],
            use_container_width=True,
            height=600,
        )

        # İndir butonu
        csv = display_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 CSV olarak indir",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    # ── TAB 5: Canlı Tahmin ──
    with tab5:
        st.subheader("🧪 Canlı Manipülasyon Testi")
        
        tab5_1, tab5_2 = st.tabs(["📝 Metin Gir", "📊 CSV Dosyası"])
        
        # TAB 5.1: Tek metin tahmini
        with tab5_1:
            st.markdown("Aşağıya bir metin girin, sistem gerçek zamanlı tahmin yapacak.")

            user_text = st.text_area(
                "Metin girin:",
                height=150,
                placeholder="Buraya test edilecek sosyal medya metnini yapıştırın...",
                key="manual_text"
            )

            if st.button("🔍 Analiz Et", type="primary", key="analyze_btn"):
                if user_text.strip():
                    with st.spinner("Analiz ediliyor..."):
                        try:
                            from csv_inference import predict_single_text
                            result = predict_single_text(user_text)

                            # Sonuç gösterimi
                            label = result['label']
                            score = result['score']

                            if label == 'MANİPÜLATİF':
                                st.error(f"🔴 **{label}** — Skor: {score:.4f} ({result['confidence']} Güven)")
                            elif label == 'ŞÜPHELİ':
                                st.warning(f"🟡 **{label}** — Skor: {score:.4f} ({result['confidence']} Güven)")
                            else:
                                st.success(f"🟢 **{label}** — Skor: {score:.4f} ({result['confidence']} Güven)")

                            # Dil bilgisi
                            st.info(f"🌍 Tespit edilen dil: **{result['language']}**")

                            # Nedenler
                            st.markdown("**Analiz Detayları:**")
                            if result['reasons']:
                                for reason in result['reasons']:
                                    st.markdown(f"- {reason}")
                            else:
                                st.markdown("- Belirgin manipülasyon sinyali yok")

                        except ImportError as e:
                            st.error(f"❌ csv_inference modülü yüklenemedi: {e}")
                        except Exception as e:
                            st.error(f"❌ Hata: {str(e)}")
                else:
                    st.warning("Lütfen bir metin girin.")
        
        # TAB 5.2: CSV dosyası tahmini
        with tab5_2:
            st.markdown("""
            **YARISMACI_TEST_GIRDISI.csv** dosyanızı yükleyin:
            - **1. Sütun**: test_id (opsiyonel)
            - **2. Sütun**: text (zorunlu)
            """)
            
            uploaded_file = st.file_uploader("CSV dosyası seçin", type="csv", key="csv_upload")
            
            if uploaded_file:
                # Dosyayı geçici olarak kaydet
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                if st.button("📊 CSV'yi İşle", type="primary", key="process_csv_btn"):
                    with st.spinner(f"🔄 {uploaded_file.name} işleniyor..."):
                        try:
                            from csv_inference import predict_from_csv
                            
                            # Sonuçları tahmin et
                            output_path = tmp_path.replace(".csv", "_results.csv")
                            df_results = predict_from_csv(tmp_path, output_path)
                            
                            if df_results is not None:
                                st.success(f"✅ {len(df_results)} satır işlendi!")
                                
                                # Özet istatistikler
                                col1, col2, col3, col4 = st.columns(4)
                                manip_count = (df_results['label'] == 'MANİPÜLATİF').sum()
                                susp_count = (df_results['label'] == 'ŞÜPHELİ').sum()
                                org_count = (df_results['label'] == 'ORGANİK').sum()
                                avg_score = df_results['score'].mean()
                                
                                col1.metric("🔴 Manipülatif", f"{manip_count:,}")
                                col2.metric("🟡 Şüpheli", f"{susp_count:,}")
                                col3.metric("🟢 Organik", f"{org_count:,}")
                                col4.metric("📈 Ort. Skor", f"{avg_score:.4f}")
                                
                                # Detaylı tablo
                                st.subheader("📋 Detaylı Sonuçlar")
                                display_cols = ['test_id', 'text', 'label', 'score', 'confidence', 'language']
                                display_cols = [c for c in display_cols if c in df_results.columns]
                                
                                st.dataframe(
                                    df_results[display_cols].head(100),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # İndir butonu
                                csv_download = df_results.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 Sonuçları CSV olarak indir",
                                    data=csv_download,
                                    file_name="YARISMACI_SONUCLAR.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("CSV işlenemedi.")
                        
                        except ImportError as e:
                            st.error(f"❌ csv_inference modülü yüklenemedi: {e}")
                        except Exception as e:
                            st.error(f"❌ Hata: {str(e)}")
                        finally:
                            # Geçici dosyayı sil
                            import os as _os
                            try:
                                _os.remove(tmp_path)
                            except:
                                pass

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        f"*Filtrelenen veri: {len(filtered_df):,} / {len(full_df):,} post*"
    )


if __name__ == '__main__':
    main()
