# -*- coding: utf-8 -*-
"""
ADIM 4 - EDA ve Veri Analitigi Raporu (A1)
Ana veri seti : Golub ALL/AML (72 ornek x 7129 gen, Affymetrix, ham sinyal + call)
Benzer veri   : GSE7186 (ALL/AML, NBM cikarilmis)

NOT (veri sizintisi): Bu betik yalnizca KESIFSEL analizdir. Burada etiket kullanilarak
secilen genler (t-testi, SNR, isi haritasi) MODELLEME ASAMASINDA KULLANILMAMALIDIR;
modelleme asamasinda tum secim/olcekleme adimlari CV katlarinin icinde yapilmalidir.

Calistirma:  python eda_adim4.py
Cikti      :  eda_cikti/figures/*.png (300 dpi), eda_cikti/tables/*.csv, eda_cikti/bulgular_ozeti.md
Gereksinim :  numpy pandas scipy scikit-learn matplotlib seaborn   (opsiyonel: umap-learn)
"""
# %% ------------------------------------------------------------------ 0. AYARLAR
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # Jupyter'de bu satiri silip plt.show() kullanabilirsiniz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --- Dosya yollari (zip'leri acip klasor yollarini buraya yazin) ---
import zipfile
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()   # betigin bulundugu klasor


def find_dir(name, must_have):
    """Veri dosyalarini ADLARINA gore bulur (klasor adi onemli degil):
    1) betikle ayni klasor, 2) BASE/<name>, 3) BASE altindaki tum alt klasorler,
    4) hicbiri yoksa BASE/<name>.zip'i acar."""
    def search():
        if all((BASE / f).exists() for f in must_have):
            return BASE
        for p in [BASE / name, *sorted(BASE.glob("**/*"))]:
            if p.is_dir() and all((p / f).exists() for f in must_have):
                return p
        return None
    found = search()
    if found is None and (BASE / f"{name}.zip").exists():
        with zipfile.ZipFile(BASE / f"{name}.zip") as z:
            z.extractall(BASE / name)
        found = search()
    if found is None:
        raise FileNotFoundError(f"{must_have} dosyalari bulunamadi. Bakilan klasor: {BASE}. "
                                f"Dosyalari betikle ayni klasore koyun.")
    print(f"  [{name}] veri klasoru: {found}")
    return found


ANA_DIR = find_dir("ana", ["actual.csv", "data_set_ALL_AML_train.csv", "data_set_ALL_AML_independent.csv"])
YAN_DIR = find_dir("yan", ["ifade_ALL_AML.csv", "etiketler_ALL_AML.csv"])
OUT = BASE / "eda_cikti"
FIG, TAB = OUT / "figures", OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# --- Opsiyonel: probe -> gen sembolu eslemesi (ortak gen sayisi icin) ---
# Her iki CSV de "probe" ve "symbol" sutunlarina sahip olmali. Yoksa None birakin.
# GSE7186 icin GEO sayfasindaki platform (GPL...) annotasyon tablosundan uretebilirsiniz.
ANNOT_GOLUB = None   # ornek: Path("annot/golub_probe_symbol.csv")
ANNOT_YAN = None     # ornek: Path("annot/gse7186_probe_symbol.csv")

# --- Analiz parametreleri ---
CLIP_LO, CLIP_HI = 20, 16000      # Golub icin alt/ust esik (yonerge 6.1)
TOP_N = 40                        # isi haritasi icin gen sayisi (yonerge: 20-50)
MAX_MISSING_YAN = 0.20            # yan veri setinde bu oranin ustunde eksigi olan gen elenir
N_PERM = 50                       # domain classifier permutasyon sayisi
SAVE_PDF = False                  # True ise vektorel PDF de kaydedilir

PAL = {"ALL": "#1f77b4", "AML": "#d62728"}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight"})

F = {}   # bulgular sozlugu (sonda otomatik ozet uretilir)


def savefig(name):
    plt.savefig(FIG / f"{name}.png")
    if SAVE_PDF:
        plt.savefig(FIG / f"{name}.pdf")
    plt.close()
    print(f"  [sekil] {name}.png")


def save_table(df, name, index=True):
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")
    print(f"  [tablo] {name}.csv")


def bh_fdr(p):
    p = np.where(np.isnan(p), 1.0, p)
    return stats.false_discovery_control(p, method="bh")


# %% ------------------------------------------------------------------ 1. VERI YUKLEME
print("=" * 70, "\n1. VERI YUKLEME VE ETIKET DOGRULAMA\n" + "=" * 70)


def load_golub():
    actual = pd.read_csv(ANA_DIR / "actual.csv")
    lab_map = actual.set_index("patient")["cancer"]

    vals, calls, splits, meta = [], [], [], None
    for split, fn in [("train", "data_set_ALL_AML_train.csv"),
                      ("test", "data_set_ALL_AML_independent.csv")]:
        df = pd.read_csv(ANA_DIR / fn)
        m = df[["Gene Description", "Gene Accession Number"]]
        if meta is None:
            meta = m
        else:   # iki dosyada gen sirasi ayni olmali
            assert (meta["Gene Accession Number"].values == m["Gene Accession Number"].values).all()
        body = df.iloc[:, 2:]
        v_cols, c_cols = body.columns[0::2], body.columns[1::2]
        # sutunlar (hasta no, call) ikilileri halinde gelir -> yapiyi dogrula
        assert all(str(c).startswith("call") for c in c_cols)
        assert not any(str(c).startswith("call") for c in v_cols)
        V = body[v_cols].apply(pd.to_numeric, errors="coerce")
        V.columns = [int(c) for c in v_cols]
        C = body[c_cols].copy()
        C.columns = V.columns
        vals.append(V)
        calls.append(C)
        splits += [(int(c), split) for c in v_cols]

    V = pd.concat(vals, axis=1)         # gen x ornek
    C = pd.concat(calls, axis=1)
    split_s = pd.Series(dict(splits), name="split")

    # ---- Etiket eslestirme dogrulamasi (hasta numarasi uzerinden) ----
    ids = list(V.columns)
    print(f"  Sutun sirasi hasta numarasina gore sirali mi? {ids == sorted(ids)}  (False beklenir)")
    assert set(ids) == set(lab_map.index), "actual.csv ile ornek kumeleri eslesmiyor!"
    assert len(ids) == len(set(ids)) == 72
    y = lab_map.loc[ids]                # sutun sirasina gore etiket (numara uzerinden)
    y.index.name = "patient"
    ct = pd.crosstab(split_s.loc[ids].values, y.values)
    print("  Split x sinif tablosu:\n", ct.to_string().replace("\n", "\n  "))
    assert ct.loc["train", "ALL"] == 27 and ct.loc["train", "AML"] == 11
    assert ct.loc["test", "ALL"] == 20 and ct.loc["test", "AML"] == 14
    print("  -> Etiket eslestirmesi dogrulandi (train 27/11, test 20/14).")

    G = V.T.copy()                      # ornek x gen
    G.columns = meta["Gene Accession Number"].values
    Cc = C.T.copy()
    Cc.columns = G.columns
    return G, Cc, y, split_s.loc[ids], meta


def load_yan():
    X = pd.read_csv(YAN_DIR / "ifade_ALL_AML.csv", index_col=0)   # gen x ornek
    lab = pd.read_csv(YAN_DIR / "etiketler_ALL_AML.csv")
    assert set(lab["GSM"]) == set(X.columns), "etiket-ornek kumesi uyumsuz"
    lab = lab.set_index("GSM").loc[X.columns]
    assert "NBM" not in set(lab["sinif"]), "NBM orneklerinin cikarilmis olmasi gerekir"
    # baslik <-> sinif tutarliligi
    ok = (lab["baslik"].str.startswith("AML") == (lab["sinif"] == "AML")).all()
    print(f"  GSE7186: baslik-sinif tutarli mi? {ok}")
    assert ok
    y = lab["sinif"].copy()
    sub = lab["baslik"].str.replace(r"\s*\d+$", "", regex=True)   # B-lineage ALL / T-cell ALL / AML
    D = X.T.copy()
    D.columns = [str(c) for c in D.columns]
    print(f"  GSE7186 sinif dagilimi: {y.value_counts().to_dict()} | alt tip: {sub.value_counts().to_dict()}")
    return D, y, sub


Gr, Gcall, y_g, split_g, meta_g = load_golub()
Yr, y_y, sub_y = load_yan()
print(f"\n  Golub : {Gr.shape[0]} ornek x {Gr.shape[1]} gen | GSE7186: {Yr.shape[0]} ornek x {Yr.shape[1]} gen")

# %% ------------------------------------------------------------------ 2. VERI KALITESI
print("\n" + "=" * 70, "\n2. VERI KALITESI\n" + "=" * 70)

# ---------- Golub ----------
A = Gr.values.astype(float)
n_inf = int(np.isinf(A).sum()); n_nan = int(np.isnan(A).sum())
n_neg = int((A < 0).sum()); n_lo = int((A < CLIP_LO).sum()); n_hi = int((A > CLIP_HI).sum())
affx = Gr.columns.str.startswith("AFFX")
dup_acc = int(pd.Series(Gr.columns).duplicated().sum())
q1, q3 = np.percentile(A, [25, 75])
n_out = int(((A < q1 - 3 * (q3 - q1)) | (A > q3 + 3 * (q3 - q1))).sum())
print(f"  Golub: INF={n_inf}, NaN={n_nan}, negatif={n_neg} (%{100*n_neg/A.size:.2f}), "
      f"<{CLIP_LO}: {n_lo} (%{100*n_lo/A.size:.2f}), >{CLIP_HI}: {n_hi}, AFFX kontrol probe={int(affx.sum())}, "
      f"tekrarlayan accession={dup_acc}")

# call (P/A/M) sutunlari
call_all = pd.Series(Gcall.values.ravel()).value_counts(normalize=True).rename("oran")
call_cls = pd.crosstab(np.repeat(y_g.values, Gcall.shape[1]), Gcall.values.ravel(), normalize="index")
print("  Call dagilimi (genel):", call_all.round(3).to_dict())
P = (Gcall == "P")
p_rate_sample = P.mean(axis=1)                       # ornek basina "Present" orani
p_rate_gene = P.mean(axis=0)                          # gen basina "Present" orani
mw = stats.mannwhitneyu(p_rate_sample[y_g == "AML"], p_rate_sample[y_g == "ALL"])
F.update(g_inf=n_inf, g_nan=n_nan, g_neg=n_neg, g_neg_pct=100 * n_neg / A.size, g_lo_pct=100 * n_lo / A.size,
         g_hi=n_hi, g_affx=int(affx.sum()), g_gene_P10=float((p_rate_gene < 0.10).mean() * 100),
         g_psample_p=float(mw.pvalue))
print(f"  Genlerin %{F['g_gene_P10']:.1f}'i orneklerin <%10'unda 'P' (nadiren ifade). "
      f"Ornek bazli P orani ALL-AML farki p={mw.pvalue:.3g}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
call_cls.plot(kind="bar", stacked=True, ax=ax[0], color=["#999999", "#f0a030", "#2ca02c"][:call_cls.shape[1]], rot=0)
ax[0].set(title="(a) Sinifa gore call (A/M/P) orani", xlabel="Sinif", ylabel="Oran")
sns.boxplot(x=y_g.values, y=p_rate_sample.values, palette=PAL, ax=ax[1], order=["ALL", "AML"])
ax[1].set(title="(b) Ornek basina 'Present' orani", xlabel="Sinif", ylabel="P orani")
ax[2].hist(p_rate_gene, bins=40, color="#555555")
ax[2].set(title="(c) Gen basina 'Present' orani", xlabel="P orani", ylabel="Gen sayisi")
plt.tight_layout(); savefig("g01_call_analizi")

# ---------- GSE7186 ----------
B = Yr.values.astype(float)
y_nan = np.isnan(B)
miss_sample = pd.Series(y_nan.mean(axis=1), index=Yr.index)
miss_gene = pd.Series(y_nan.mean(axis=0), index=Yr.columns)
Bn = B[~y_nan]
q1y, q3y = np.percentile(Bn, [25, 75])
n_out_y = int(((Bn < q1y - 3 * (q3y - q1y)) | (Bn > q3y + 3 * (q3y - q1y))).sum())
mw_y = stats.mannwhitneyu(miss_sample[y_y == "AML"], miss_sample[y_y == "ALL"])
F.update(y_nan_pct=100 * y_nan.mean(), y_genes_complete=int((miss_gene == 0).sum()),
         y_genes_ok=int((miss_gene <= MAX_MISSING_YAN).sum()), y_min=float(Bn.min()), y_max=float(Bn.max()),
         y_med=float(np.median(Bn)), y_neg_pct=100 * (Bn < 0).mean(), y_miss_p=float(mw_y.pvalue))
print(f"  GSE7186: INF={int(np.isinf(B).sum())}, NaN=%{F['y_nan_pct']:.2f}, min={Bn.min():.2f}, max={Bn.max():.2f}, "
      f"medyan={np.median(Bn):.3f}, negatif=%{F['y_neg_pct']:.1f}, uc deger(3*IQR)={n_out_y}")
print(f"  Hic eksigi olmayan gen={F['y_genes_complete']}, eksik<=%{int(MAX_MISSING_YAN*100)} gen={F['y_genes_ok']}; "
      f"ornek bazli eksiklik ALL-AML farki p={mw_y.pvalue:.3g}")
print("  -> Degerler ~0 merkezli ve negatif: ham sinyal degil, ZATEN log-oran/normalize olcek. Log donusumu UYGULANMAZ.")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(miss_gene, bins=50, color="#555555"); ax[0].axvline(MAX_MISSING_YAN, c="r", ls="--")
ax[0].set(title="(a) Gen basina eksik oran", xlabel="Eksik oran", ylabel="Gen sayisi", yscale="log")
sns.boxplot(x=y_y.values, y=miss_sample.values, palette=PAL, ax=ax[1], order=["ALL", "AML"])
ax[1].set(title="(b) Ornek basina eksik oran", xlabel="Sinif", ylabel="Eksik oran")
ax[2].hist(Bn, bins=120, color="#555555", density=True)
ax[2].set(title="(c) Gozlenen ifade degerleri", xlabel="Ifade (log-oran)", ylabel="Yogunluk")
plt.tight_layout(); savefig("y01_veri_kalitesi")

# Tablo 1: veri kalitesi ozeti
tq = pd.DataFrame({
    "Golub (ana)": {
        "Ornek": Gr.shape[0], "Gen/prob": Gr.shape[1], "INF": n_inf, "NaN": n_nan,
        "Negatif deger (%)": round(100 * n_neg / A.size, 3), f"<{CLIP_LO} (%)": round(100 * n_lo / A.size, 2),
        f">{CLIP_HI} (adet)": n_hi, "AFFX kontrol probe": int(affx.sum()),
        "Uc deger 3xIQR (adet)": n_out, "Olcek": "Ham sinyal (Affymetrix) + call"},
    "GSE7186 (yan)": {
        "Ornek": Yr.shape[0], "Gen/prob": Yr.shape[1], "INF": int(np.isinf(B).sum()), "NaN": int(y_nan.sum()),
        "Negatif deger (%)": round(100 * (Bn < 0).mean(), 3), f"<{CLIP_LO} (%)": "-",
        f">{CLIP_HI} (adet)": "-", "AFFX kontrol probe": "-",
        "Uc deger 3xIQR (adet)": n_out_y, "Olcek": "Log-oran (zaten donusturulmus), eksik degerli"}})
save_table(tq.astype(str), "t01_veri_kalitesi")

# %% ------------------------------------------------------------------ 3. BETIMSEL ISTATISTIK + LOG DONUSUMU
print("\n" + "=" * 70, "\n3. BETIMSEL ISTATISTIK, SINIF DAGILIMI, LOG DONUSUMU\n" + "=" * 70)

# --- Sinif dagilimi ---
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a_, (title, ser) in zip(ax, [("(a) Golub - tum (n=72)", y_g),
                                 ("(b) Golub - egitim / bagimsiz test", None),
                                 ("(c) GSE7186 (n=%d)" % len(y_y), y_y)]):
    if ser is not None:
        c = ser.value_counts().reindex(["ALL", "AML"])
        bars = a_.bar(c.index, c.values, color=[PAL[k] for k in c.index])
        for b_, v_ in zip(bars, c.values):
            a_.text(b_.get_x() + b_.get_width() / 2, v_ + 0.5, f"{v_} (%{100*v_/c.sum():.0f})", ha="center")
        a_.set(title=title, ylabel="Ornek sayisi", ylim=(0, c.max() * 1.18))
    else:
        ct = pd.crosstab(split_g, y_g).loc[["train", "test"], ["ALL", "AML"]]
        ct.plot(kind="bar", ax=a_, color=[PAL["ALL"], PAL["AML"]], rot=0)
        for cont in a_.containers:
            a_.bar_label(cont)
        a_.set(title=title, xlabel="", ylabel="Ornek sayisi", ylim=(0, ct.values.max() * 1.18))
        a_.set_xticklabels(["Egitim (38)", "Test (34)"])
plt.tight_layout(); savefig("01_sinif_dagilimi")

# --- Golub: log donusumu oncesi / sonrasi ---
Gc = Gr.clip(CLIP_LO, CLIP_HI)                  # alt/ust esik (ornek-bagimsiz sabit esik -> sizinti yok)
Gl = np.log2(Gc)
raw_flat, log_flat = Gr.values.ravel(), Gl.values.ravel()
skew_raw, skew_log = stats.skew(raw_flat), stats.skew(log_flat)
F.update(skew_raw=float(skew_raw), skew_log=float(skew_log))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(np.clip(raw_flat, -1000, 20000), bins=150, color="#555555")
ax[0].set(yscale="log", title=f"(a) Ham deger (carpiklik={skew_raw:.1f})", xlabel="Sinyal", ylabel="Frekans (log)")
ax[1].hist(np.clip(raw_flat, CLIP_LO, CLIP_HI), bins=150, color="#777777")
ax[1].set(yscale="log", title=f"(b) {CLIP_LO}-{CLIP_HI} araligina kirpilmis", xlabel="Sinyal", ylabel="Frekans (log)")
ax[2].hist(log_flat, bins=100, color="#2ca02c")
ax[2].set(title=f"(c) Kirpma + log2 (carpiklik={skew_log:.2f})", xlabel="log2(sinyal)", ylabel="Frekans")
plt.tight_layout(); savefig("02_golub_log_oncesi_sonrasi")

# ornek bazli dagilimlar (sinifa ve split'e gore) - ham vs log
smp = pd.DataFrame({"sinif": y_g.values, "split": split_g.values,
                    "medyan_ham": Gr.median(axis=1).values, "medyan_log2": Gl.median(axis=1).values,
                    "std_ham": Gr.std(axis=1).values, "std_log2": Gl.std(axis=1).values})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a_, col, t in zip(ax, ["medyan_ham", "medyan_log2", "std_ham", "std_log2"],
                      ["(a) Ornek medyani (ham)", "(b) Ornek medyani (log2)", "(c) Ornek std (ham)", "(d) Ornek std (log2)"]):
    sns.boxplot(data=smp, x="sinif", y=col, order=["ALL", "AML"], palette=PAL, ax=a_, showfliers=False)
    sns.stripplot(data=smp, x="sinif", y=col, order=["ALL", "AML"], color="k", size=2.5, alpha=.6, ax=a_)
    a_.set(title=t, xlabel="Sinif", ylabel="")
plt.tight_layout(); savefig("03_golub_ornek_ozetleri")

# --- GSE7186: ifade dagilimi (zaten log-olcek) ---
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k in ["ALL", "AML"]:
    ax[0].hist(Yr[y_y == k].values[~np.isnan(Yr[y_y == k].values)], bins=120, alpha=.55, density=True,
               color=PAL[k], label=k)
ax[0].set(title="(a) Sinifa gore ifade dagilimi", xlabel="Ifade (log-oran)", ylabel="Yogunluk"); ax[0].legend()
ym = pd.DataFrame({"sinif": y_y.values, "medyan": Yr.median(axis=1).values})
sns.boxplot(data=ym, x="sinif", y="medyan", order=["ALL", "AML"], palette=PAL, ax=ax[1], showfliers=False)
sns.stripplot(data=ym, x="sinif", y="medyan", order=["ALL", "AML"], color="k", size=2.5, alpha=.6, ax=ax[1])
ax[1].set(title="(b) Ornek medyani", xlabel="Sinif", ylabel="Medyan ifade")
plt.tight_layout(); savefig("y02_ifade_dagilimi")

# --- Tablo 2: betimsel istatistikler (sinifa gore) ---
def describe_block(Xdf, y, name):
    rows = []
    for k in ["ALL", "AML", "Tum"]:
        sub_ = Xdf if k == "Tum" else Xdf[y.values == k]
        v = sub_.values.astype(float)
        v = v[~np.isnan(v)]
        rows.append({"Veri": name, "Sinif": k, "n": sub_.shape[0], "Ortalama": v.mean(), "Std": v.std(),
                     "Medyan": np.median(v), "Min": v.min(), "Max": v.max(),
                     "Carpiklik": stats.skew(v), "Basiklik": stats.kurtosis(v)})
    return pd.DataFrame(rows)

t2 = pd.concat([describe_block(Gr, y_g, "Golub ham"), describe_block(Gl, y_g, "Golub log2"),
                describe_block(Yr, y_y, "GSE7186 (log-oran)")], ignore_index=True)
save_table(t2.round(3), "t02_betimsel_istatistik", index=False)
print(t2.round(2).to_string(index=False))

# %% ------------------------------------------------------------------ 4. EDA ICIN ON ISLEME
print("\n" + "=" * 70, "\n4. KESIFSEL ON ISLEME (etiket KULLANILMAZ)\n" + "=" * 70)

# Golub: AFFX kontrol probelerini cikar, klasik varyans filtresi (max/min>5 ve max-min>500), log2, z-skor
Gc_nc = Gc.loc[:, ~affx]
mx, mn = Gc_nc.max(), Gc_nc.min()
keep_g = (mx / mn > 5) & ((mx - mn) > 500)
Gl_f = np.log2(Gc_nc.loc[:, keep_g])
print(f"  Golub: {Gr.shape[1]} -> AFFX cikarildi {Gc_nc.shape[1]} -> varyans filtresi sonrasi {Gl_f.shape[1]} gen")
Zg = pd.DataFrame(StandardScaler().fit_transform(Gl_f), index=Gl_f.index, columns=Gl_f.columns)

# GSE7186: eksik orani yuksek genleri cikar, gen medyani ile doldur, z-skor
keep_y = miss_gene[miss_gene <= MAX_MISSING_YAN].index
Y_f = Yr[keep_y]
Y_imp = Y_f.fillna(Y_f.median())
print(f"  GSE7186: {Yr.shape[1]} -> eksik<=%{int(MAX_MISSING_YAN*100)} filtresi sonrasi {Y_f.shape[1]} gen "
      f"(kalan eksik degerler gen medyaniyla dolduruldu - yalniz EDA icin)")
Zy = pd.DataFrame(StandardScaler().fit_transform(Y_imp), index=Y_imp.index, columns=Y_imp.columns)
F.update(g_genes_after=Gl_f.shape[1], y_genes_after=Y_f.shape[1])

# %% ------------------------------------------------------------------ 5. BOYUT INDIRGEME (PCA / t-SNE / UMAP)
print("\n" + "=" * 70, "\n5. BOYUT INDIRGEME\n" + "=" * 70)


def run_pca(Z, name, y, extra=None, extra_name=None, extra_pal=None):
    npc_all = min(Z.shape[0] - 1, 100)
    pca = PCA(n_components=npc_all, random_state=SEED).fit(Z)
    S = pca.transform(Z)
    ev, cev = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cev, 0.90) + 1) if cev[-1] >= 0.90 else None
    F[f"n90_{name}"] = n90
    npc = min(30, npc_all)   # grafikte ilk 30 PC
    print(f"  {name}: PC1=%{100*ev[0]:.1f}, PC2=%{100*ev[1]:.1f}, ilk 2 PC=%{100*cev[1]:.1f}, "
          f"%90 varyans icin PC sayisi={n90 if n90 else f'>{npc_all}'}")
    sil2 = silhouette_score(S[:, :2], y.values)
    sil10 = silhouette_score(S[:, :10], y.values)
    fig, ax = plt.subplots(1, 3 if extra is not None else 2, figsize=(14 if extra is not None else 10, 4))
    ax[0].bar(range(1, npc + 1), 100 * ev[:npc], color="#4c72b0", label="Bireysel")
    ax[0].plot(range(1, npc + 1), 100 * cev[:npc], "o-", c="k", ms=3, label="Kumulatif")
    ax[0].set(title="(a) Scree grafigi", xlabel="Temel bilesen", ylabel="Aciklanan varyans (%)"); ax[0].legend()
    for k in ["ALL", "AML"]:
        m = (y.values == k)
        ax[1].scatter(S[m, 0], S[m, 1], c=PAL[k], label=f"{k} (n={m.sum()})", s=40, edgecolor="k", lw=.4)
    ax[1].set(title=f"(b) PCA - sinif (silhouette={sil2:.2f})",
              xlabel=f"PC1 (%{100*ev[0]:.1f})", ylabel=f"PC2 (%{100*ev[1]:.1f})"); ax[1].legend()
    if extra is not None:
        for k, c in extra_pal.items():
            m = (extra.values == k)
            ax[2].scatter(S[m, 0], S[m, 1], c=c, label=f"{k} (n={m.sum()})", s=40, edgecolor="k", lw=.4)
        ax[2].set(title=f"(c) PCA - {extra_name}", xlabel=f"PC1 (%{100*ev[0]:.1f})", ylabel=f"PC2 (%{100*ev[1]:.1f})")
        ax[2].legend()
    plt.tight_layout(); savefig(f"04_pca_{name}")
    return S, ev, cev, sil2, sil10


Sg, evg, cevg, sil2g, sil10g = run_pca(Zg, "golub", y_g, split_g, "veri bolumu",
                                       {"train": "#2ca02c", "test": "#ff7f0e"})
Sy, evy, cevy, sil2y, sil10y = run_pca(Zy, "gse7186", y_y, sub_y, "alt tip",
                                       {"B-lineage ALL": "#1f77b4", "T-cell ALL": "#17becf", "AML": "#d62728"})
F.update(g_pc1=100 * evg[0], g_pc2=100 * evg[1], g_pc12=100 * cevg[1], g_sil2=sil2g, g_sil10=sil10g,
         y_pc1=100 * evy[0], y_pc2=100 * evy[1], y_pc12=100 * cevy[1], y_sil2=sil2y, y_sil10=sil10y)
save_table(pd.DataFrame({"Golub_PC_var(%)": 100 * evg[:20], "Golub_kum(%)": 100 * cevg[:20],
                         "GSE7186_PC_var(%)": 100 * evy[:20], "GSE7186_kum(%)": 100 * cevy[:20]},
                        index=range(1, 21)).round(2), "t03_pca_varyans")

# t-SNE (+ UMAP varsa)
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    print("  (umap-learn kurulu degil -> yalnizca t-SNE. Kurmak icin: pip install umap-learn)")


def run_embed(Z, y, name):
    n = Z.shape[0]
    Z50 = PCA(n_components=min(50, n - 1), random_state=SEED).fit_transform(Z)
    perps = [5, 15, 30] if n < 100 else [10, 30, 50]
    ncol = len(perps) + (1 if HAS_UMAP else 0)
    fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4))
    sils = {}
    for a_, p_ in zip(ax, perps):
        E = TSNE(2, perplexity=min(p_, (n - 1) / 3), init="pca", learning_rate="auto", random_state=SEED).fit_transform(Z50)
        for k in ["ALL", "AML"]:
            m = y.values == k
            a_.scatter(E[m, 0], E[m, 1], c=PAL[k], label=k, s=36, edgecolor="k", lw=.4)
        sils[f"tsne_p{p_}"] = silhouette_score(E, y.values)
        a_.set(title=f"t-SNE (perplexity={p_}, sil={sils[f'tsne_p{p_}']:.2f})", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax[0].legend()
    if HAS_UMAP:
        E = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(Z50)
        for k in ["ALL", "AML"]:
            m = y.values == k
            ax[-1].scatter(E[m, 0], E[m, 1], c=PAL[k], label=k, s=36, edgecolor="k", lw=.4)
        sils["umap"] = silhouette_score(E, y.values)
        ax[-1].set(title=f"UMAP (sil={sils['umap']:.2f})", xlabel="UMAP 1", ylabel="UMAP 2")
    plt.suptitle(f"{name}: t-SNE{' / UMAP' if HAS_UMAP else ''} (girdi: z-skor -> 50 PC)", y=1.02)
    plt.tight_layout(); savefig(f"05_tsne_{name}")
    return sils


sil_g, sil_y = run_embed(Zg, y_g, "golub"), run_embed(Zy, y_y, "gse7186")
F.update(g_tsne_sil=max(sil_g.values()), y_tsne_sil=max(sil_y.values()))

# %% ------------------------------------------------------------------ 6. AYIRT EDICI GENLER
print("\n" + "=" * 70, "\n6. AYIRT EDICI GENLER (YALNIZ KESIF - modellemede kullanilmaz)\n" + "=" * 70)


def de_table(X, y, labels, names=None):
    """Welch t-testi, Mann-Whitney, BH-FDR, ortalama fark (AML-ALL), Cohen d, Golub SNR."""
    is1 = (y.values == "AML")
    a, b = X.values[~is1].astype(float), X.values[is1].astype(float)
    t, p = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit", axis=0)
    # Mann-Whitney icin eksik degerleri gen medyaniyla doldur
    Xm = X.fillna(X.median()).values.astype(float)
    pm = stats.mannwhitneyu(Xm[is1], Xm[~is1], axis=0).pvalue
    m1, m0 = np.nanmean(b, 0), np.nanmean(a, 0)
    s1, s0 = np.nanstd(b, 0, ddof=1), np.nanstd(a, 0, ddof=1)
    n1, n0 = (~np.isnan(b)).sum(0), (~np.isnan(a)).sum(0)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    out = pd.DataFrame({"gen": X.columns, "ort_AML": m1, "ort_ALL": m0, "fark_AML_ALL": m1 - m0,
                        "t": t, "p": p, "FDR": bh_fdr(p), "p_MWU": pm, "FDR_MWU": bh_fdr(pm),
                        "cohen_d": (m1 - m0) / sp, "SNR": (m1 - m0) / (s1 + s0)}).set_index("gen")
    if names is not None:
        out.insert(0, "aciklama", names.reindex(out.index))
    return out


# --- Golub (log2, tum 72 ornek, AFFX haric, varyans filtresi OLMADAN) ---
Gl_all = np.log2(Gc_nc)
desc = meta_g.set_index("Gene Accession Number")["Gene Description"]
de_g = de_table(Gl_all, y_g, None, desc)
# --- GSE7186 (eksik<=%20 genler) ---
de_y = de_table(Y_f, y_y, None)

for nm, de in [("golub", de_g), ("gse7186", de_y)]:
    de.sort_values("p").to_csv(TAB / f"t04_de_tum_{nm}.csv", encoding="utf-8-sig")
    top = de.reindex(de["t"].abs().sort_values(ascending=False).index).head(TOP_N)
    top.round(4).to_csv(TAB / f"t05_en_iyi_{TOP_N}_gen_{nm}.csv", encoding="utf-8-sig")
    print(f"  [tablo] t04_de_tum_{nm}.csv, t05_en_iyi_{TOP_N}_gen_{nm}.csv")

sig_g = (de_g["FDR"] < 0.05); sig_y = (de_y["FDR"] < 0.05)
big_g = sig_g & (de_g["fark_AML_ALL"].abs() > 1); big_y = sig_y & (de_y["fark_AML_ALL"].abs() > 1)
F.update(g_sig=int(sig_g.sum()), g_sig_pct=100 * sig_g.mean(), g_sig_fc=int(big_g.sum()),
         y_sig=int(sig_y.sum()), y_sig_pct=100 * sig_y.mean(), y_sig_fc=int(big_y.sum()),
         g_top=de_g["t"].abs().sort_values(ascending=False).head(5).index.tolist(),
         y_top=de_y["t"].abs().sort_values(ascending=False).head(5).index.tolist())
print(f"  Golub  : FDR<0.05 gen = {sig_g.sum()} (%{F['g_sig_pct']:.1f}); ek |fark|>1 (log2) = {big_g.sum()}")
print(f"  GSE7186: FDR<0.05 gen = {sig_y.sum()} (%{F['y_sig_pct']:.1f}); ek |fark|>1 = {big_y.sum()}")

# Volcano
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a_, de, nm in zip(ax, [de_g, de_y], ["(a) Golub (log2)", "(b) GSE7186 (log-oran)"]):
    x = de["fark_AML_ALL"]; yy = -np.log10(de["p"].clip(lower=1e-300))
    sg = de["FDR"] < 0.05
    a_.scatter(x[~sg], yy[~sg], s=5, c="#bbbbbb", label="FDR>=0.05")
    a_.scatter(x[sg & (x > 0)], yy[sg & (x > 0)], s=6, c=PAL["AML"], label="AML'de yuksek (FDR<0.05)")
    a_.scatter(x[sg & (x < 0)], yy[sg & (x < 0)], s=6, c=PAL["ALL"], label="ALL'de yuksek (FDR<0.05)")
    a_.set(title=f"{nm} - Volkan grafigi", xlabel="Ortalama fark (AML - ALL)", ylabel="-log10(p)")
    a_.legend(markerscale=2, fontsize=8)
plt.tight_layout(); savefig("06_volcano")

# Top-N gen: kutu/strip grafigi (ilk 8 gen)
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for r, (X_, y_, de, nm, dsc) in enumerate([(Gl_all, y_g, de_g, "Golub", desc), (Y_f, y_y, de_y, "GSE7186", None)]):
    for c, g in enumerate(de["t"].abs().sort_values(ascending=False).index[:8]):
        d_ = pd.DataFrame({"sinif": y_.values, "deger": X_[g].values})
        sns.boxplot(data=d_, x="sinif", y="deger", order=["ALL", "AML"], palette=PAL, ax=axes[r, c], showfliers=False)
        sns.stripplot(data=d_, x="sinif", y="deger", order=["ALL", "AML"], color="k", size=2, alpha=.5, ax=axes[r, c])
        lab = str(g) if dsc is None else f"{g}\n{str(dsc.get(g, ''))[:22]}"
        axes[r, c].set(title=lab, xlabel="", ylabel=nm if c == 0 else "")
        axes[r, c].title.set_fontsize(7.5)
plt.tight_layout(); savefig("07_en_iyi_8_gen_kutu")

# Isi haritasi + hiyerarsik kumeleme (etiket-bilgili ilk TOP_N gen) -----------------
def heat(X, y, de, name, desc_map=None):
    top = de["t"].abs().sort_values(ascending=False).index[:TOP_N]
    sub_ = X[top]
    order = np.argsort(y.values)             # sinif renkleri icin
    lut = y.map(PAL)
    lab = [(f"{g} | {str(desc_map.get(g, ''))[:28]}" if desc_map is not None else str(g)) for g in top]
    sub_plot = sub_.copy(); sub_plot.columns = lab
    cg = sns.clustermap(sub_plot.T, z_score=0, method="ward", metric="euclidean", cmap="vlag", center=0, vmin=-3, vmax=3,
                        col_colors=lut.values, figsize=(12, 10), xticklabels=False,
                        yticklabels=True, cbar_kws={"label": "Gen basina z-skor"},
                        dendrogram_ratio=(0.12, 0.08))
    cg.ax_heatmap.set_xlabel(f"Ornekler (n={len(y)}); ust serit: mavi=ALL, kirmizi=AML")
    cg.ax_heatmap.set_ylabel("")
    cg.ax_heatmap.tick_params(axis="y", labelsize=7)
    cg.fig.suptitle(f"{name}: en ayirt edici {TOP_N} gen (|t| siralamasi) - hiyerarsik kumeleme (Ward)", y=1.01)
    plt.savefig(FIG / f"08_isi_haritasi_{name}.png"); plt.close("all")
    print(f"  [sekil] 08_isi_haritasi_{name}.png")
    # 2 kume + saflik
    L = linkage(StandardScaler().fit_transform(sub_.values), method="ward")
    cl = fcluster(L, 2, criterion="maxclust")
    return adjusted_rand_score(y.values, cl)


ari_top_g = heat(Gl_all, y_g, de_g, "golub", desc)
ari_top_y = heat(Y_imp, y_y, de_y, "gse7186")
print(f"  Top-{TOP_N} gen ile Ward 2-kume ARI: Golub={ari_top_g:.2f}, GSE7186={ari_top_y:.2f} "
      f"(etiket-bilgili secim oldugu icin iyimser!)")

# Denetimsiz hiyerarsik kumeleme (tum filtrelenmis genler, etiket KULLANILMAZ) -------
def unsup_cluster(Z, y, name, extra=None):
    L = linkage(Z.values, method="ward")
    cl = fcluster(L, 2, criterion="maxclust")
    ari = adjusted_rand_score(y.values, cl)
    tab = pd.crosstab(cl, y.values)
    fig, ax = plt.subplots(figsize=(14 if Z.shape[0] > 100 else 12, 4.2))
    dn = dendrogram(L, ax=ax, no_labels=False, labels=[f"{i}" for i in Z.index], leaf_font_size=6,
                    color_threshold=0, above_threshold_color="#444444")
    pos = {str(i): y.iloc[k] for k, i in enumerate(Z.index)}
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(PAL[pos[lbl.get_text()]])
    ax.set(title=f"{name}: denetimsiz Ward kumeleme (ARI={ari:.2f}); etiket rengi: mavi=ALL, kirmizi=AML",
           ylabel="Ward mesafesi")
    plt.tight_layout(); savefig(f"09_dendrogram_{name}")
    print(f"  {name} denetimsiz 2-kume ARI={ari:.2f}; kume x sinif:\n    " + tab.to_string().replace("\n", "\n    "))
    return ari, tab


ari_g, tab_cl_g = unsup_cluster(Zg, y_g, "golub")
ari_y, tab_cl_y = unsup_cluster(Zy, y_y, "gse7186")
F.update(g_ari_unsup=ari_g, y_ari_unsup=ari_y, g_ari_top=ari_top_g, y_ari_top=ari_top_y)

# Golub: egitim vs bagimsiz test'te gen siralamasi tekrarlanabilir mi? ------------------
tr, te = (split_g == "train").values, (split_g == "test").values
de_tr = de_table(Gl_all[tr], y_g[tr], None)
de_te = de_table(Gl_all[te], y_g[te], None)
ok_ = np.isfinite(de_tr["t"]) & np.isfinite(de_te["t"])   # kirpma sonrasi sabit kalan genlerde t=NaN
rho = stats.spearmanr(de_tr["t"][ok_], de_te["t"][ok_]).correlation
top_tr = set(de_tr["t"].abs().sort_values(ascending=False).index[:50])
top_te = set(de_te["t"].abs().sort_values(ascending=False).index[:50])
ov = len(top_tr & top_te)
F.update(g_rho_tt=float(rho), g_overlap50=ov)
print(f"  Golub egitim vs test t-istatistigi Spearman rho={rho:.2f}; ilk-50 gen ortusmesi={ov}/50")
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(de_tr["t"], de_te["t"], s=4, c="#555555", alpha=.5)
ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
ax.set(xlabel="t (egitim, n=38)", ylabel="t (bagimsiz test, n=34)", title=f"Gen etkisinin tekrarlanabilirligi (rho={rho:.2f})")
plt.tight_layout(); savefig("10_golub_train_test_t")

# %% ------------------------------------------------------------------ 7. BATCH ETKISI (GOLUB EGITIM vs TEST)
print("\n" + "=" * 70, "\n7. BATCH ETKISI: GOLUB EGITIM vs BAGIMSIZ TEST\n" + "=" * 70)
sp_lab = split_g.values
sil_split = silhouette_score(Sg[:, :10], sp_lab)
print(f"  PCA(10 PC) uzayinda silhouette: sinif={sil10g:.2f}, veri bolumu(split)={sil_split:.2f}")

# (a) siniflar ici gen bazli train-test farki
rows = []
for k in ["ALL", "AML"]:
    m = (y_g.values == k)
    xa, xb = Gl_all[m & tr].values, Gl_all[m & te].values
    t_, p_ = stats.ttest_ind(xa, xb, equal_var=False, axis=0)
    q_ = bh_fdr(p_)
    rows.append({"Sinif": k, "n_egitim": int((m & tr).sum()), "n_test": int((m & te).sum()),
                 "FDR<0.05 gen": int((q_ < 0.05).sum()), "oran(%)": round(100 * (q_ < 0.05).mean(), 2)})
tb = pd.DataFrame(rows); save_table(tb, "t06_batch_gen_testi", index=False); print(tb.to_string(index=False))
F.update(g_batch_genes=int(tb["FDR<0.05 gen"].sum()))

# (b) domain classifier: gen ifadesinden egitim/test ayirt edilebiliyor mu?
def domain_auc(Z, lab, n_perm=N_PERM):
    """Egitim/test ayirt edilebilir mi? Kat bazli AUC ortalamasi (katlar arasi karar esigi
    farklarindan kaynaklanan yapay <0.5 AUC'yi onlemek icin) + permutasyon p-degeri."""
    lab = np.asarray(lab)
    top_var = Z.var().sort_values(ascending=False).index[:1000]
    X_ = Z[top_var].values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))

    def fold_auc(l, seed):
        aucs = []
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_, l):
            pipe.fit(X_[tr_i], l[tr_i] == "test")
            aucs.append(roc_auc_score(l[te_i] == "test", pipe.decision_function(X_[te_i])))
        return float(np.mean(aucs))

    auc = fold_auc(lab, SEED)
    rng = np.random.RandomState(SEED)
    null = np.array([fold_auc(rng.permutation(lab), SEED) for _ in range(n_perm)])
    return auc, (np.sum(null >= auc) + 1) / (n_perm + 1)


rows = []
for nm, m in [("Tum ornekler", np.ones(len(y_g), bool)), ("Yalniz ALL", y_g.values == "ALL"), ("Yalniz AML", y_g.values == "AML")]:
    auc, pv = domain_auc(Zg[m], sp_lab[m])
    rows.append({"Altküme": nm, "n": int(m.sum()), "Egitim/Test AUC (kat ort., 5-kat CV)": round(auc, 3), "Permutasyon p": round(pv, 3)})
td = pd.DataFrame(rows); save_table(td, "t07_domain_classifier", index=False); print(td.to_string(index=False))
F.update(g_dom_auc_all=float(td.iloc[0, 2]), g_dom_auc_all_p=float(td.iloc[0, 3]),
         g_dom_auc_aml=float(td.iloc[2, 2]), g_dom_auc_all_only=float(td.iloc[1, 2]))

# (c) gorsel: PCA (sinif x split) + ornek ozetleri
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
mk = {"train": "o", "test": "^"}
for k in ["ALL", "AML"]:
    for s_ in ["train", "test"]:
        m = (y_g.values == k) & (sp_lab == s_)
        ax[0].scatter(Sg[m, 0], Sg[m, 1], c=PAL[k], marker=mk[s_], s=48, edgecolor="k", lw=.5,
                      label=f"{k} - {'egitim' if s_ == 'train' else 'test'} (n={m.sum()})")
ax[0].set(title=f"(a) PCA: sinif (renk) x bolum (sekil), sil_split={sil_split:.2f}",
          xlabel=f"PC1 (%{100*evg[0]:.1f})", ylabel=f"PC2 (%{100*evg[1]:.1f})"); ax[0].legend(fontsize=7)
smp["bolum"] = np.where(smp["split"] == "train", "Egitim", "Test")
sns.boxplot(data=smp, x="sinif", y="medyan_log2", hue="bolum", order=["ALL", "AML"], ax=ax[1], showfliers=False,
            palette={"Egitim": "#2ca02c", "Test": "#ff7f0e"})
ax[1].set(title="(b) Ornek medyani (log2)", xlabel="Sinif", ylabel="Medyan log2")
sns.boxplot(data=smp, x="sinif", y="std_log2", hue="bolum", order=["ALL", "AML"], ax=ax[2], showfliers=False,
            palette={"Egitim": "#2ca02c", "Test": "#ff7f0e"})
ax[2].set(title="(c) Ornek std (log2)", xlabel="Sinif", ylabel="Std log2")
plt.tight_layout(); savefig("11_batch_etkisi_golub")

# %% ------------------------------------------------------------------ 8. IKI VERI SETI ARASI FARKLAR
print("\n" + "=" * 70, "\n8. GOLUB vs GSE7186 DAGILIM FARKLARI\n" + "=" * 70)

# her ornek kendi icinde z-skorlanir (olcek farkini ortadan kaldirip SEKIL farkina bakmak icin)
def per_sample_z(X):
    v = X.values.astype(float)
    return (v - np.nanmean(v, 1, keepdims=True)) / np.nanstd(v, 1, keepdims=True)

zg, zy = per_sample_z(Gl_all).ravel(), per_sample_z(Y_f).ravel()
zy = zy[~np.isnan(zy)]
ks = stats.ks_2samp(zg[np.random.RandomState(SEED).choice(len(zg), 200000, replace=False)],
                    zy[np.random.RandomState(SEED).choice(len(zy), 200000, replace=False)])
F.update(ks_stat=float(ks.statistic), skew_gz=float(stats.skew(zg)), skew_yz=float(stats.skew(zy)))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(Gl_all.values.ravel(), bins=120, alpha=.6, density=True, label="Golub (log2 sinyal)", color="#2ca02c")
ax[0].hist(Bn, bins=120, alpha=.6, density=True, label="GSE7186 (log-oran)", color="#9467bd")
ax[0].set(title="(a) Orijinal olcekte ifade", xlabel="Ifade", ylabel="Yogunluk"); ax[0].legend(fontsize=8)
sns.kdeplot(zg[::50], ax=ax[1], label=f"Golub (carpiklik={F['skew_gz']:.2f})", color="#2ca02c")
sns.kdeplot(zy[::50], ax=ax[1], label=f"GSE7186 (carpiklik={F['skew_yz']:.2f})", color="#9467bd")
ax[1].set(title=f"(b) Ornek-ici z-skor (KS={ks.statistic:.2f})", xlabel="z-skor", ylabel="Yogunluk", xlim=(-5, 5)); ax[1].legend(fontsize=8)
gs, ys = Gl_all.std(), Y_f.std()
ax[2].hist(gs, bins=60, alpha=.6, density=True, label="Golub", color="#2ca02c")
ax[2].hist(ys, bins=60, alpha=.6, density=True, label="GSE7186", color="#9467bd")
ax[2].set(title="(c) Gen basina standart sapma", xlabel="Std", ylabel="Yogunluk"); ax[2].legend(fontsize=8)
plt.tight_layout(); savefig("12_iki_veri_dagilim")

# Tablo 3: yan yana ozet
t3 = pd.DataFrame({
    "Golub (ana)": {"Kaynak": "Golub ve ark., Science 1999 (DOI: 10.1126/science.286.5439.531)",
                    "Ornek (n)": Gr.shape[0], "Ozellik (p)": Gr.shape[1], "ALL / AML": f"{(y_g=='ALL').sum()} / {(y_g=='AML').sum()}",
                    "AML orani (%)": round(100 * (y_g == 'AML').mean(), 1), "Olcek": "Ham Affymetrix sinyali (+call)",
                    "Eksik (%)": 0.0, "Egitim/Test": "38 / 34"},
    "GSE7186 (yan)": {"Kaynak": "GEO GSE7186",
                      "Ornek (n)": Yr.shape[0], "Ozellik (p)": Yr.shape[1], "ALL / AML": f"{(y_y=='ALL').sum()} / {(y_y=='AML').sum()}",
                      "AML orani (%)": round(100 * (y_y == 'AML').mean(), 1), "Olcek": "Log-oran (normalize)",
                      "Eksik (%)": round(F["y_nan_pct"], 2), "Egitim/Test": "yok (CV ile)"}})
save_table(t3, "t08_veri_seti_ozeti"); print(t3.to_string())

# Ortak gen sayisi (opsiyonel annotasyon)
if ANNOT_GOLUB is not None and ANNOT_YAN is not None:
    ag, ay = pd.read_csv(ANNOT_GOLUB), pd.read_csv(ANNOT_YAN)
    sg_, sy_ = set(ag["symbol"].dropna().astype(str).str.upper()), set(ay["symbol"].dropna().astype(str).str.upper())
    F["common_genes"] = len(sg_ & sy_)
    print(f"  Ortak gen sembolu: {F['common_genes']} (Golub {len(sg_)}, GSE7186 {len(sy_)})")
else:
    print("  Ortak gen sayisi HESAPLANMADI: GSE7186 satirlari sayisal ID_REF (probe) ve Golub'da Affymetrix accession var.\n"
          "  Probe->gen sembolu eslemesi icin GEO'daki platform (GPL) annotasyonu gerekir; ANNOT_* ayarlarina bakin.")

# %% ------------------------------------------------------------------ 9. OTOMATIK BULGU OZETI
print("\n" + "=" * 70, "\n9. BULGULARIN NUMARALI OZETI\n" + "=" * 70)
bat = "belirgin bir batch/dagilim kaymasi var" if F["g_dom_auc_all"] > 0.75 and F["g_dom_auc_all_p"] < 0.05 \
    else "belirgin bir batch kaniti zayif"
lines = [
    f"1. Etiket eslestirmesi hasta numarasi uzerinden dogrulandi; Golub egitim 27 ALL/11 AML, test 20 ALL/14 AML "
    f"(AML orani %{100*(y_g=='AML').mean():.0f}); GSE7186'da {(y_y=='ALL').sum()} ALL/{(y_y=='AML').sum()} AML "
    f"(AML orani %{100*(y_y=='AML').mean():.0f}) -> her iki veri seti dengesiz; stratified bolme ve dengesizlige duyarli metrikler (Balanced Acc., F1, PR-AUC) gerekir.",
    f"2. Golub'da INF={F['g_inf']}, NaN={F['g_nan']}; ancak degerlerin %{F['g_neg_pct']:.2f}'i negatif, %{F['g_lo_pct']:.1f}'i <{CLIP_LO}; "
    f"{F['g_affx']} AFFX kontrol probesi var -> alt/ust esik kirpma ve kontrol probelerinin cikarilmasi onerilir.",
    f"3. Ham Golub verisi saga carpik (carpiklik={F['skew_raw']:.1f}); kirpma + log2 sonrasi {F['skew_log']:.2f}'ye iner -> log2 donusumu gerekli.",
    f"4. Golub'da genlerin %{F['g_gene_P10']:.0f}'i orneklerin <%10'unda 'Present' -> dusuk ifadeli/varyansli gen filtresi mantikli "
    f"(klasik filtre sonrasi {F['g_genes_after']} gen).",
    f"5. GSE7186 zaten log-oran olcegindedir (min={F['y_min']:.1f}, max={F['y_max']:.1f}, medyan={F['y_med']:.2f}) ve "
    f"%{F['y_nan_pct']:.1f} eksik degerlidir; sadece {F['y_genes_complete']} gen tam doludur -> imputasyon (medyan/KNN) ve eksik-oran filtresi gerekir; "
    f"eksiklik orani siniflar arasinda {'farkli (p=%.3g)' % F['y_miss_p'] if F['y_miss_p'] < 0.05 else 'anlamli farkli degil (p=%.2f)' % F['y_miss_p']}. "
    "Imputasyon CV katlarinin icinde yapilmalidir.",
    f"6. PCA: Golub'da ilk 2 PC varyansin %{F['g_pc12']:.1f}'ini (PC1 %{F['g_pc1']:.1f}), GSE7186'da %{F['y_pc12']:.1f}'ini aciklar; "
    f"sinif silhouette (10 PC): Golub={F['g_sil10']:.2f}, GSE7186={F['y_sil10']:.2f}; denetimsiz Ward 2-kume ARI: Golub={F['g_ari_unsup']:.2f}, "
    f"GSE7186={F['y_ari_unsup']:.2f}.",
    f"7. Ayirt edici gen: FDR<0.05 gen sayisi Golub'da {F['g_sig']} (%{F['g_sig_pct']:.1f}), GSE7186'da {F['y_sig']} (%{F['y_sig_pct']:.1f}) -> "
    "sinyal cok sayida genede dagilmistir (p >> n problemi icin filtre/gomulu secim ve regularizasyon anlamli). En iyi genler (|t|): "
    f"Golub {', '.join(map(str, F['g_top']))}; GSE7186 {', '.join(map(str, F['y_top']))}.",
    f"8. Golub'da egitim ve test t-istatistikleri arasinda Spearman rho={F['g_rho_tt']:.2f}, ilk-50 gen ortusmesi={F['g_overlap50']}/50 -> "
    + ("ayirt edici sinyal bagimsiz ornekte buyuk olcude tekrarlanir." if F["g_rho_tt"] > 0.5
   else "gen etkileri egitim ve test arasinda sinirli tutarlilik gosterir; bu, kucuk n ve olasi batch farkina isaret edebilir."),
    f"9. Batch: siniflar icinde egitim/test arasi FDR<0.05 gen sayisi toplam {F['g_batch_genes']}; egitim/test domain-classifier AUC "
    f"(tum)={F['g_dom_auc_all']:.2f} (perm. p={F['g_dom_auc_all_p']:.3f}), yalniz AML={F['g_dom_auc_aml']:.2f} -> {bat}. "
    "Sabit 38/34 bolmede dis gecerlilik iyimser/kotumser sapabilir; tum 72 ornekte nested CV ile karsilastirilmalidir.",
    f"10. Iki veri seti farkli platform/olcektedir (ham sinyal vs log-oran; ornek-ici z-skor KS={F['ks_stat']:.2f}); "
    + (f"ortak gen sayisi {F['common_genes']}. " if "common_genes" in F else "probe->gen eslemesi olmadan ortak gen sayisi bilinmiyor. ")
    + "Ayni 3 yontem ayni protokolle her iki veri setine ayri ayri uygulanmali (dogrudan birlestirme/transfer yapilmamali).",
    "11. Modelleme cikarimlari: (i) olcekleme (z-skor) mesafe/dogrusal modeller (SVM, LR, kNN) icin gerekli, agac tabanlilar icin gereksiz; "
    "(ii) imputasyon, filtre, ozellik secimi ve olcekleme yalniz egitim katlarinda fit edilmeli; (iii) class_weight='balanced' + stratified "
    "bolme; (iv) Accuracy yerine Balanced Accuracy/F1/ROC-AUC/PR-AUC birlikte raporlanmali; (v) etiketle secilmis EDA genleri modellemede kullanilmamali.",
]
txt = "# EDA bulgularinin numarali ozeti\n\n" + "\n\n".join(lines) + "\n"
(OUT / "bulgular_ozeti.md").write_text(txt, encoding="utf-8")
print(txt)
print("Bitti. Ciktilar:", OUT.resolve())
