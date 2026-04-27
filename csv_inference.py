###############################################################################
# BATCH CSV INFERENCE — 500 SATIRLIK TEST VERİSİ
# Giriş: original_text sütunu içeren CSV
# Çıkış: skor, karar, açıklama sütunları eklenmiş CSV
#
# Kullanım:
#   python csv_inference.py girdi.csv cikti.csv
#
# Gerekli dosyalar (C:\datathon\ altında):
#   ensemble_config.pkl
#   xgb_seeds_session1.pkl
#   xgb_seeds_session2.pkl
###############################################################################

import os
import re
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect

SAVE_DIR = r"C:\datathon"

THRESHOLD_HIGH  = 0.70
THRESHOLD_MID_H = 0.40
THRESHOLD_MID_L = 0.20

# ══════════════════════════════════════════
# 1. MODEL YÜKLEME
# ══════════════════════════════════════════
print("Modeller yükleniyor...")
t0 = time.time()

try:
    with open(os.path.join(SAVE_DIR, "ensemble_config.pkl"), "rb") as f:
        config = pickle.load(f)

    top_seeds = config["top_seeds"]
    weights   = [float(w) for w in config["weights"]]
    feat_cols = config["feat_cols"]

    all_session_models = {}
    for fname in ["xgb_seeds_session1.pkl", "xgb_seeds_session2.pkl"]:
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                sess = pickle.load(f)
            for seed, result in sess.items():
                if "model" in result:
                    all_session_models[seed] = result["model"]

    models = [all_session_models[s] for s in top_seeds]
    print(f"  ✓ {len(models)} XGBoost modeli yüklendi (seeds: {top_seeds})")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"  ✓ bge-m3 yüklendi ({device})")
    print(f"  Toplam yükleme: {time.time()-t0:.1f}s")
    
    MODELS_LOADED = True

except Exception as e:
    print(f"  ⚠ Model yükleme hatası: {e}")
    print(f"  ⚠ Ön filtreleme ve embedding ile devam edilecek")
    models = []
    embed_model = None
    weights = []
    feat_cols = []
    MODELS_LOADED = False

# ══════════════════════════════════════════
# 2. ÖN FİLTRE
# ══════════════════════════════════════════
BOX_CHARS = re.compile(
    r'[▬◗◖⎐⊵⊴←→─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬▀▄▌▐░▒▓■□▪▫◄►◀▶◁▷]'
)
SCAM_DOMAINS = re.compile(
    r'https?://(?:www\.)?('
    r'alfiaa\d*\.site|dailysolly\.eu|xsfa\d*\.xyz'
    r')', re.IGNORECASE
)
TRIPLE_CHAR = re.compile(r'([a-zA-Z])\1{2,}')
NATURAL_DOUBLES = {
    "coffee","book","cool","feel","free","keep","look","need","pass","room",
    "see","seem","soon","too","tool","week","well","will","wood","good","food",
    "took","school","meet","been","deep","feet","green","tree","three","wheel",
    "sweet","teeth","speed","sleep","steel","street","cheese","sheep"
}

def has_deliberate_typo(word):
    if len(word) < 5: return False
    clean = word.lower().strip(".,!?;:'\"")
    if clean in NATURAL_DOUBLES: return False
    return bool(re.search(r'([a-zA-Z])\1', clean))

def pre_filter(text):
    box_found = BOX_CHARS.findall(text)
    if len(box_found) >= 2:
        return 1.0, 0.0, "[Ön Filtre] Dekoratif kutu/çerçeve karakterleri tespit edildi — spam kupon şablonu."
    scam = SCAM_DOMAINS.search(text)
    if scam:
        return 1.0, 0.0, f"[Ön Filtre] Bilinen dolandırıcılık domain'i: {scam.group(0)[:50]}"
    # Bahis direkt manipülatif
    gambling_kw = ["bahis","casino","jackpot","bahisci","teslabahis","prizmabet","norabahis","orisbet"]
    if any(kw in text.lower() for kw in gambling_kw):
        found = [kw for kw in gambling_kw if kw in text.lower()]
        return 1.0, 0.0, f"[Ön Filtre] Yasadışı bahis/kumar reklamı tespit edildi: {', '.join(found[:3])}"
    triple = TRIPLE_CHAR.findall(text)
    if triple:
        return 0.0, 0.25, f"[Ön Filtre] Kasıtlı karakter tekrarı (bot imzası). Skor +0.25."
    typo_words = [w for w in text.split() if has_deliberate_typo(w)]
    if len(typo_words) >= 3:
        return 0.0, 0.25, f"[Ön Filtre] Kasıtlı yazım hataları: {typo_words[:4]}. Skor +0.25."
    return 0.0, 0.0, None

# ══════════════════════════════════════════
# 3. FEATURE ÇIKARIMI
# ══════════════════════════════════════════
def extract_features(text):
    words = text.split()
    wc = len(words)
    tl = len(text)
    sents = max(len(re.split(r'[.!?]+', text)), 1)
    return {
        "text_length":         tl,
        "word_count":          wc,
        "avg_word_length":     np.mean([len(w) for w in words]) if words else 0,
        "char_per_word":       tl / max(wc, 1),
        "hashtag_count":       len(re.findall(r'#\w+', text)),
        "mention_count":       len(re.findall(r'@\w+', text)),
        "url_in_text_count":   len(re.findall(r'https?://\S+', text)),
        "emoji_count":         len(re.findall(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]', text)),
        "caps_ratio":          sum(1 for c in text if c.isupper()) / max(tl, 1),
        "exclamation_count":   text.count('!'),
        "question_mark_count": text.count('?'),
        "ellipsis_count":      text.count('...'),
        "newline_count":       text.count('\n'),
        "repeated_char_ratio": sum(1 for i in range(1, tl) if text[i] == text[i-1]) / max(tl, 1),
        "unique_word_ratio":   len(set(w.lower() for w in words)) / max(wc, 1),
        "sentence_count":      sents,
        "avg_sentence_length": wc / sents,
        "digit_ratio":         sum(1 for c in text if c.isdigit()) / max(tl, 1),
        "special_char_ratio":  sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(tl, 1),
    }

# ══════════════════════════════════════════
# 4. SEBEPLENDIRME
# ══════════════════════════════════════════
def generate_explanation(score, feat, text):
    signals = []
    counter_signals = []
    tl = text.lower()

    if feat["hashtag_count"] >= 5:
        signals.append(f"Aşırı hashtag ({feat['hashtag_count']} adet) — koordineli dağıtıma işaret")
    elif feat["hashtag_count"] >= 2:
        signals.append(f"Çoklu hashtag ({feat['hashtag_count']})")
    if feat["mention_count"] >= 4:
        signals.append(f"Toplu etiketleme ({feat['mention_count']} mention) — bot etiketleme yöntemi")
    elif feat["mention_count"] >= 2:
        signals.append(f"Çoklu mention ({feat['mention_count']})")
    if feat["url_in_text_count"] >= 2:
        signals.append(f"Çoklu URL ({feat['url_in_text_count']}) — trafik yönlendirme/dolandırıcılık ihtimali")
    elif feat["url_in_text_count"] == 1:
        signals.append("Harici bağlantı içeriyor")
    if feat["emoji_count"] >= 6:
        signals.append(f"Yoğun emoji ({feat['emoji_count']}) — kripto pump şablonu görünümü")
    elif feat["emoji_count"] >= 3:
        signals.append(f"Çoklu emoji ({feat['emoji_count']})")
    if feat["caps_ratio"] > 0.35:
        signals.append(f"Aşırı büyük harf ({feat['caps_ratio']:.0%}) — dikkat çekme amaçlı")
    if feat["exclamation_count"] >= 4:
        signals.append(f"Aşırı ünlem ({feat['exclamation_count']}) — aciliyet hissi yaratma")

    crypto_kw = ["airdrop","token","mint","nft","pump","moon","hodl","presale",
                 "testnet","listing","gem","100x","earn big","don't miss",
                 "to the moon","passive income","free gifts","referral","launchpool"]
    found_kw = [kw for kw in crypto_kw if kw in tl]
    if len(found_kw) >= 2:
        signals.append(f"Kripto pump/spam kampanyası: {', '.join(found_kw[:4])}")
    elif found_kw:
        signals.append(f"Kripto/spam anahtar kelimesi: {found_kw[0]}")

    gambling_kw = ["bahis","casino","bonus","oyna","kazan","bet","jackpot"]
    found_gamb = [kw for kw in gambling_kw if kw in tl]
    if found_gamb:
        signals.append(f"Bahis/kumar reklamı: {', '.join(found_gamb[:3])}")

    engage_kw = ["follow me","follow back","retweet","like and","join now","dm me","accept my invite"]
    found_eng = [kw for kw in engage_kw if kw in tl]
    if found_eng:
        signals.append(f"Etkileşim farming: '{found_eng[0]}'")

    coord_kw = ["would bring","would love to see","let's make it happen",
                "community is thriving","community is growing"]
    if any(kw in tl for kw in coord_kw):
        signals.append("Koordineli kampanya şablon dili")

    organic_kw = ["i think","i feel","honestly","lol","lmao","ngl","bence","açıkçası"]
    found_org = [kw for kw in organic_kw if kw in tl]
    if found_org:
        counter_signals.append(f"Doğal konuşma dili ('{found_org[0]}')")
    if feat["hashtag_count"]==0 and feat["mention_count"]==0 and feat["url_in_text_count"]==0:
        counter_signals.append("Tanıtım işareti yok")
    if feat["word_count"] > 40 and feat["unique_word_ratio"] > 0.80:
        counter_signals.append("Yüksek kelime çeşitliliği")

    if score >= THRESHOLD_HIGH:
        return "MANİPÜLATİF: " + ("; ".join(signals) if signals else "Semantik yapı manipülatif örüntülerle örtüşüyor")
    elif score >= THRESHOLD_MID_H:
        p = []
        if signals: p.append("Şüpheli: " + "; ".join(signals))
        if counter_signals: p.append("Organik: " + "; ".join(counter_signals))
        return " | ".join(p) or "Karışık sinyaller"
    elif score >= THRESHOLD_MID_L:
        p = []
        if counter_signals: p.append("Organik: " + "; ".join(counter_signals))
        if signals: p.append("Zayıf sinyal: " + "; ".join(signals))
        return " | ".join(p) or "Ağırlıklı organik"
    else:
        return "ORGANİK: " + ("; ".join(counter_signals) if counter_signals else "Manipülasyon sinyali yok")

# ══════════════════════════════════════════
# 5. DİL TESPİTİ
# ══════════════════════════════════════════
def detect_lang(text):
    try:
        return detect(text[:500])
    except:
        return "unknown"

# ══════════════════════════════════════════
# 6. KOORDİNELİ KAMPANYA TESPİTİ
# ══════════════════════════════════════════
def detect_coordinated(texts, threshold=0.85, min_cluster=3):
    """
    Embedding benzerliği ile koordineli kampanya kümeleri bul.
    3+ benzer metin → hepsini manipülatif işaretle.
    """
    print("Koordineli kampanya analizi (embedding benzerliği)...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    coordinated = set()
    clusters = []

    visited = set()
    for i in range(len(texts)):
        if i in visited:
            continue
        neighbors = np.where(sim_matrix[i] >= threshold)[0]
        if len(neighbors) >= min_cluster - 1:
            cluster = [i] + list(neighbors)
            cluster = list(set(cluster))
            clusters.append(cluster)
            for idx in cluster:
                coordinated.add(idx)
                visited.add(idx)

    print(f"  {len(clusters)} koordineli küme bulundu, {len(coordinated)} post etkilendi")
    return coordinated, clusters, embeddings

# ══════════════════════════════════════════
# 7. BATCH INFERENCE
# ══════════════════════════════════════════
def run_batch_inference(input_path, output_path):
    print(f"\nGirdi: {input_path}")

    # CSV veya Parquet oku
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # Metin sütununu bul
    text_col = None
    for col in ["original_text", "text", "content", "message"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]
    print(f"  Metin sütunu: '{text_col}'")
    print(f"  Toplam satır: {len(df):,}")

    texts = df[text_col].fillna("").astype(str).tolist()

    # Koordineli kampanya tespiti
    coord_indices, coord_clusters, all_embeddings = detect_coordinated(texts)
    print(f"  Koordineli: {len(coord_indices)} post")

    # Her satır için inference
    results = []
    t0 = time.time()

    for i, text in enumerate(texts):
        if not text.strip():
            results.append({"skor": 0.0, "karar": "ORGANİK", "aciklama": "Boş metin", "dil": "unknown"})
            continue

        # Koordineli kampanya → direkt manipülatif
        if i in coord_indices:
            feat = extract_features(text)
            lang = detect_lang(text)
            cluster_size = 0
            for cl in coord_clusters:
                if i in cl:
                    cluster_size = len(cl)
                    break
            results.append({
                "skor": 1.0,
                "karar": "MANİPÜLATİF",
                "aciklama": f"[Koordineli Kampanya] Bu metin {cluster_size} farklı kayıtta benzer şekilde görünüyor (cosine similarity ≥ 0.85). Koordineli bot kampanyası.",
                "dil": lang
            })
            continue

        # Ön filtre
        score_override, score_bonus, filter_reason = pre_filter(text)

        if score_override == 1.0:
            feat = extract_features(text)
            lang = detect_lang(text)
            results.append({
                "skor": 1.0,
                "karar": "MANİPÜLATİF",
                "aciklama": filter_reason,
                "dil": lang
            })
            continue

        # XGBoost inference
        feat = extract_features(text)
        emb = all_embeddings[i]
        row = {**feat, **{f"emb_{j}": float(emb[j]) for j in range(len(emb))}}
        X = np.array([[row.get(col, 0.0) for col in feat_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        xgb_score = sum(
            float(w) * model.predict_proba(X)[0][1]
            for model, w in zip(models, weights)
        )
        score = min(1.0, xgb_score + score_bonus)

        if score >= THRESHOLD_HIGH:
            label = "MANİPÜLATİF"
        elif score >= THRESHOLD_MID_H:
            label = "ŞÜPHELİ MANİPÜLATİF"
        elif score >= THRESHOLD_MID_L:
            label = "ŞÜPHELİ ORGANİK"
        else:
            label = "ORGANİK"

        explanation = generate_explanation(score, feat, text)
        if score_bonus > 0 and filter_reason:
            explanation = filter_reason + " | " + explanation

        lang = detect_lang(text)

        results.append({
            "skor": round(score, 4),
            "karar": label,
            "aciklama": explanation,
            "dil": lang
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            remaining = (len(texts) - i - 1) / speed
            print(f"  [{i+1:,}/{len(texts):,}] {speed:.1f} post/s | kalan: {remaining:.0f}s")

    # Sonuçları birleştir
    df_results = pd.DataFrame(results)
    df_out = pd.concat([df.reset_index(drop=True), df_results], axis=1)

    # Kaydet
    if output_path.endswith(".csv"):
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    n_manip = (df_results["karar"] == "MANİPÜLATİF").sum()
    n_suspect = (df_results["karar"] == "ŞÜPHELİ MANİPÜLATİF").sum()
    n_organic = (df_results["karar"].isin(["ORGANİK", "ŞÜPHELİ ORGANİK"])).sum()

    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE TAMAMLANDI")
    print(f"{'='*60}")
    print(f"  Toplam: {len(df):,} | Süre: {elapsed:.1f}s")
    print(f"  🔴 Manipülatif:          {n_manip:,}")
    print(f"  🟠 Şüpheli Manipülatif:  {n_suspect:,}")
    print(f"  🟢 Organik:              {n_organic:,}")
    print(f"  Koordineli küme:         {len(coord_clusters)}")
    print(f"  Çıktı: {output_path}")

# ══════════════════════════════════════════
# 8. SINGLE TEXT PREDICTION (Dashboard için)
# ══════════════════════════════════════════

class ManipulationPredictor:
    """Streamlit dashboard için cached model wrapper"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        # Modeller zaten yüklü (globalda)
        self._initialized = True
    
    def predict_single(self, text):
        """Tek metin için tahmin yap"""
        if not text.strip():
            return {
                "label": "ORGANİK",
                "score": 0.0,
                "confidence": "Yüksek",
                "reasons": ["Boş metin"],
                "language": "unknown"
            }
        
        # Ön filtre
        score_override, score_bonus, filter_reason = pre_filter(text)
        
        if score_override == 1.0:
            lang = detect_lang(text)
            return {
                "label": "MANİPÜLATİF",
                "score": 1.0,
                "confidence": "Yüksek",
                "reasons": [filter_reason] if filter_reason else ["Ön filtrede tespit edildi"],
                "language": lang
            }
        
        # Modeller yüklenemezse ön filtreye dayalı karar ver
        if not MODELS_LOADED:
            lang = detect_lang(text)
            if filter_reason:
                return {
                    "label": "ŞÜPHELİ",
                    "score": 0.5,
                    "confidence": "Düşük",
                    "reasons": [filter_reason, "XGBoost modelleri yüklü değil - sınırlı analiz"],
                    "language": lang
                }
            else:
                return {
                    "label": "ORGANİK",
                    "score": 0.0,
                    "confidence": "Düşük",
                    "reasons": ["XGBoost modelleri yüklü değil - yalnızca ön filtre uygulandı"],
                    "language": lang
                }
        
        # Feature çıkarımı
        feat = extract_features(text)
        emb = embed_model.encode([text])[0]
        
        row = {**feat, **{f"emb_{j}": float(emb[j]) for j in range(len(emb))}}
        X = np.array([[row.get(col, 0.0) for col in feat_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # XGBoost ensemble
        xgb_score = sum(
            float(w) * model.predict_proba(X)[0][1]
            for model, w in zip(models, weights)
        )
        score = min(1.0, xgb_score + score_bonus)
        
        # Label belirle
        if score >= THRESHOLD_HIGH:
            label = "MANİPÜLATİF"
            confidence = "Yüksek"
        elif score >= THRESHOLD_MID_H:
            label = "ŞÜPHELİ"
            confidence = "Orta"
        else:
            label = "ORGANİK"
            confidence = "Yüksek"
        
        # Açıklama
        explanation = generate_explanation(score, feat, text)
        reasons = explanation.split("|")
        reasons = [r.strip() for r in reasons if r.strip()]
        
        lang = detect_lang(text)
        
        return {
            "label": label,
            "score": round(float(score), 4),
            "confidence": confidence,
            "reasons": reasons[:5],
            "language": lang
        }


def predict_single_text(text):
    """Streamlit tarafından çağrılan fonksiyon"""
    predictor = ManipulationPredictor()
    return predictor.predict_single(text)


def predict_from_csv(csv_path, output_path=None):
    """YARISMACI_TEST_GIRDISI.csv dosyasından tahmin yap"""
    print(f"\n📥 CSV okunuyor: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Dosya bulunamadı: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    if 'text' not in df.columns:
        print("❌ 'text' sütunu bulunamadı!")
        return None
    
    print(f"✓ {len(df)} satır okundu")
    
    results = []
    predictor = ManipulationPredictor()
    
    for idx, row in df.iterrows():
        text = str(row.get('text', '')).strip()
        result = predictor.predict_single(text)
        result['test_id'] = row.get('test_id', idx)
        results.append(result)
        
        if (idx + 1) % 50 == 0:
            print(f"  ✓ {idx + 1}/{len(df)} işlendi")
    
    df_results = pd.DataFrame(results)
    
    # Sütunları yeniden düzenle
    cols_order = ['test_id', 'label', 'score', 'confidence', 'language', 'reasons']
    cols_order = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[cols_order]
    
    df_out = pd.concat([df.reset_index(drop=True), df_results], axis=1)
    
    if output_path:
        df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ Sonuçlar kaydedildi: {output_path}")
        print(f"\nÖzet:")
        print(f"  🔴 Manipülatif: {(df_results['label'] == 'MANİPÜLATİF').sum()}")
        print(f"  🟡 Şüpheli:     {(df_results['label'] == 'ŞÜPHELİ').sum()}")
        print(f"  🟢 Organik:     {(df_results['label'] == 'ORGANİK').sum()}")
    
    return df_out


# ══════════════════════════════════════════
# 9. ÇALIŞTIR
# ══════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_batch_inference(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        inp = sys.argv[1]
        out = inp.rsplit(".", 1)[0] + "_scored.csv"
        run_batch_inference(inp, out)
    else:
        print("Kullanım:")
        print("  python csv_inference.py girdi.csv cikti.csv")
        print("  python csv_inference.py girdi.csv")
        print("  (çıktı adı belirtilmezse girdi_scored.csv olur)")
