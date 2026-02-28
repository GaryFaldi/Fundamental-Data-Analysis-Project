import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Air Quality Analytics Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# --- DATA LOADING ---
# @st.cache_data
# def load_data():
#     df = pd.read_csv("cleaned_air_quality_data.csv")
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     return df

merged_df = pd.read_csv("merged_air_quality_data.csv", encoding="latin1")
cleaned_df = pd.read_csv("cleaned_air_quality_data.csv", encoding="latin1")
treshold_df = pd.read_csv("treshold_air_quality_data.csv", encoding="latin1")


st.title("Data Analytics Project - Asah")
st.sidebar.title("Page Navigation")
st.sidebar.markdown("Kadek Gary Faldi - garyfaldi1@gmail.com")
st.sidebar.markdown("Select a page to navigate through the app.")
page = st.sidebar.selectbox("Choose your page", ["Home", "Data Cleaning", "Data Exploration", "Analisis Lanjutan"])

if page == "Home":
    st.header("Welcome to the Data Analytics Project - Asah")
    st.subheader("Dataset Preview")

    PAGE_SIZE = 1000
    TOTAL_ROWS = len(merged_df)
    TOTAL_PAGES = (TOTAL_ROWS - 1) // PAGE_SIZE + 1

    if "page" not in st.session_state:
        st.session_state.page = 1

    start = (st.session_state.page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, TOTAL_ROWS)

    st.dataframe(
        merged_df.iloc[start:end],
        hide_index=True
    )


    st.caption(
        f"Showing rows {start + 1:,} – {end:,} of {TOTAL_ROWS:,}"
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅ Prev", disabled=st.session_state.page == 1):
            st.session_state.page -= 1
            st.rerun()

    with col2:
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>"
                f"Page {st.session_state.page} / {TOTAL_PAGES}"
                f"</p>",
                unsafe_allow_html=True
            )

            page = st.number_input(
                "",
                min_value=1,
                max_value=TOTAL_PAGES,
                value=st.session_state.page,
                step=1,
                label_visibility="collapsed"
            )

            if page != st.session_state.page:
                st.session_state.page = page
                st.rerun()

    with col3:
        if st.button("Next ➡", disabled=st.session_state.page == TOTAL_PAGES):
            st.session_state.page += 1
            st.rerun()

    st.subheader("Dataset Features")
    st.write(merged_df.columns.tolist())

    st.subheader("Dataset Information")
    summary = pd.DataFrame({
    "dtype": merged_df.dtypes.astype(str),
    "missing": merged_df.isnull().sum(),
    "unique": merged_df.nunique(),
    })
    st.dataframe(summary)

    st.subheader("Descriptive Statistics")
    st.dataframe(merged_df.describe(include=[np.number]))

elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning Process")

    st.markdown("""
    Tahap *data cleaning* merupakan proses fundamental dalam data analytics untuk memastikan 
    kualitas, konsistensi, dan reliabilitas data sebelum dilakukan eksplorasi dan analisis lanjutan.  

    Data kualitas udara sering mengandung nilai hilang, kesalahan pencatatan, dan anomali ekstrem 
    yang dapat menyebabkan bias analisis apabila tidak ditangani dengan tepat.
    """)

    with st.expander("1️⃣ Konversi Tipe Data & Pembentukan Fitur Waktu"):
        st.markdown("""
        **Apa yang dilakukan:**
        - Menggabungkan kolom `year`, `month`, `day`, dan `hour` menjadi satu variabel waktu `datetime`.

        **Tujuan:**
        - Memudahkan analisis berbasis waktu (harian, bulanan, musiman, tahunan)
        - Mendukung visualisasi time-series dan tren polusi.

        **Manfaat:**
        - Deteksi pola musiman
        - Analisis jam sibuk
        - Tren jangka panjang yang lebih akurat
        """)

    with st.expander("2️⃣ Penanganan Missing Values"):
        st.markdown("""
        **Apa yang dilakukan:**
        - Mengisi nilai kosong pada variabel polutan dan meteorologi menggunakan mean atau median.

        **Alasan metode ini dipilih:**
        - Mean cocok untuk distribusi relatif normal
        - Median lebih robust terhadap outlier ekstrem

        **Tujuan:**
        - Menghindari hilangnya informasi akibat penghapusan baris data
        - Menjaga stabilitas statistik dataset

        **Manfaat:**
        - Analisis korelasi tetap valid
        - Model prediktif tidak terganggu oleh data kosong
        """)

    with st.expander("3️⃣ Penghapusan Duplikasi dan Outlier"):
        st.markdown("""
        **Apa yang dilakukan:**
        - Menghapus baris data yang tercatat ganda
        - Mengeliminasi nilai ekstrem tidak realistis terutama pada PM2.5

        **Tujuan:**
        - Mengurangi distorsi statistik
        - Mencegah lonjakan palsu pada visualisasi

        **Manfaat:**
        - Distribusi data lebih representatif
        - Korelasi meteorologi lebih akurat
        - Tren polusi mencerminkan kondisi nyata
        """)

    st.subheader("📊 Dampak Data Cleaning")

    st.markdown("""
    Setelah proses pembersihan:

    ✔ Data menjadi konsisten secara temporal  
    ✔ Nilai ekstrem yang menyesatkan telah dikurangi  
    ✔ Analisis pola musiman dan korelasi cuaca menjadi lebih reliabel  
    ✔ Siap digunakan untuk eksplorasi lanjutan dan pemodelan prediktif  
    """)

    st.info(f"Jumlah data setelah pembersihan: {cleaned_df.shape[0]:,} baris.")

    st.subheader("🗂 Cleaned Dataset Preview")

    PAGE_SIZE = 1000
    TOTAL_ROWS = len(cleaned_df)
    TOTAL_PAGES = (TOTAL_ROWS - 1) // PAGE_SIZE + 1

    if "clean_page" not in st.session_state:
        st.session_state.clean_page = 1

    start = (st.session_state.clean_page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, TOTAL_ROWS)

    st.dataframe(
        cleaned_df.iloc[start:end],
        hide_index=True
    )

    st.caption(f"Showing rows {start + 1:,} – {end:,} of {TOTAL_ROWS:,}")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅ Prev", disabled=st.session_state.clean_page == 1, key="clean_prev"):
            st.session_state.clean_page -= 1
            st.rerun()

    with col2:
        st.markdown(
            f"<p style='text-align:center; font-weight:600;'>"
            f"Page {st.session_state.clean_page} / {TOTAL_PAGES}"
            f"</p>",
            unsafe_allow_html=True
        )

        page_input = st.number_input(
            "",
            min_value=1,
            max_value=TOTAL_PAGES,
            value=st.session_state.clean_page,
            step=1,
            label_visibility="collapsed",
            key="clean_input"
        )

        if page_input != st.session_state.clean_page:
            st.session_state.clean_page = page_input
            st.rerun()

    with col3:
        if st.button("Next ➡", disabled=st.session_state.clean_page == TOTAL_PAGES, key="clean_next"):
            st.session_state.clean_page += 1
            st.rerun()


# --- 3️⃣ PAGE: DATA EXPLORATION ---
elif page == "Data Exploration":
    st.title("📊 Exploratory Data Analysis")

    stasiun_list = cleaned_df['station'].unique().tolist()
    options = ["Semua Stasiun"] + stasiun_list
    selected_station = st.multiselect(
        "Pilih Stasiun:",
        options,
        default=["Semua Stasiun"],
        max_selections=12
    )

    if len(selected_station) == 0:
        st.error("Pilih setidaknya 1 stasiun untuk menganalisis data dan pola.")
        st.stop()

    if "Semua Stasiun" in selected_station and len(selected_station) > 1:
        st.warning("Jika memilih 'Semua Stasiun', tidak boleh memilih stasiun lain.")
        st.stop()

    if len(selected_station) > 12:
        st.warning("Maksimal hanya boleh memilih 12 stasiun.")
        st.stop()

    cleaned_df['datetime'] = pd.to_datetime(cleaned_df['datetime'], errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['datetime'])

    if "Semua Stasiun" in selected_station:
        filtered_df = cleaned_df
    else:
        filtered_df = cleaned_df[cleaned_df['station'].isin(selected_station)]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Tren & Perbandingan", 
        "🌦️ Pola Meteorologi", 
        "📅 Variasi Waktu", 
        "🔬 Analisis Korelasi",
        "🕒 Fluktuasi Harian"
    ])

    # ==============================
    # TAB 1: TREN & PERBANDINGAN
    # ==============================
    with tab1:
        st.subheader("Tren Konsentrasi PM2.5 Tahunan")
        yearly_df = filtered_df.set_index('datetime').resample('YE').mean(numeric_only=True).reset_index()
        years_data = yearly_df['datetime'].dt.year
        values_data = yearly_df['PM2.5']
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(years_data, values_data, alpha=0.6, color='skyblue', label='Rata-rata PM2.5')
        ax1.plot(years_data, values_data, marker='o', linewidth=2, color='orange', label='Tren PM2.5')
        for x, y in zip(years_data, values_data):
            ax1.text(x, y + 1, f'{y:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_title('Tren Tahunan PM2.5')
        st.pyplot(fig1)

        st.divider()

        st.subheader("Perbandingan PM2.5 vs PM10 per Stasiun")
        station_mean = filtered_df.groupby('station')[['PM2.5', 'PM10']].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        x_idx = np.arange(len(station_mean['station']))
        width = 0.35
        ax2.bar(x_idx - width/2, station_mean['PM2.5'], width, label='PM2.5', color='#3498db')
        ax2.bar(x_idx + width/2, station_mean['PM10'], width, label='PM10', color="#fd7200")
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(station_mean['station'], rotation=45)
        ax2.legend()
        st.pyplot(fig2)

    # ==============================
    # TAB 2: POLA METEOROLOGI
    # ==============================
    with tab2:
        col_rain, col_wind = st.columns(2)
        
        temp_df = filtered_df.copy()
        temp_df['month'] = temp_df['datetime'].dt.month
        
        with col_rain:
            st.subheader("Pola Curah Hujan")
            monthly_rain = temp_df.groupby('month')['RAIN'].mean()
            fig3, ax3 = plt.subplots()
            ax3.bar(monthly_rain.index, monthly_rain.values, color="#005eff")
            ax3.set_title("Rata-rata Curah Hujan (mm/jam)")
            st.pyplot(fig3)

        with col_wind:
            st.subheader("Pola Kecepatan Angin")
            monthly_wind = temp_df.groupby('month')['WSPM'].mean()
            fig4, ax4 = plt.subplots()
            ax4.bar(monthly_wind.index, monthly_wind.values, color="#00fffb")
            ax4.set_title("Rata-rata Kecepatan Angin (m/s)")
            st.pyplot(fig4)

    # ==============================
    # TAB 3: VARIASI WAKTU (BULANAN & MUSIMAN)
    # ==============================
    with tab3:
        st.subheader("Variasi Bulanan Polutan (Z-Score)")
        pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        monthly_mean = temp_df.groupby('month')[pollutants].mean()
        scaler = StandardScaler()
        monthly_scaled = pd.DataFrame(scaler.fit_transform(monthly_mean), index=monthly_mean.index, columns=monthly_mean.columns)
        
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        for col in monthly_scaled.columns:
            ax5.plot(monthly_scaled.index, monthly_scaled[col], marker='o', label=col)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        st.pyplot(fig5)

        st.divider()

        st.subheader("Variasi Musiman Polutan")
        # Logika season tetap sama seperti kode Anda
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Spring'
            elif month in [6, 7, 8]: return 'Summer'
            else: return 'Autumn'
        
        temp_df['season'] = temp_df['month'].apply(get_season)
        seasonal_mean = temp_df.groupby("season")[pollutants].mean().reindex(["Winter", "Spring", "Summer", "Autumn"])
        
        if not seasonal_mean.dropna().empty:
            seasonal_scaled = pd.DataFrame(scaler.fit_transform(seasonal_mean), index=seasonal_mean.index, columns=seasonal_mean.columns)
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            for col in seasonal_scaled.columns:
                ax6.plot(seasonal_scaled.index, seasonal_scaled[col], marker="o", label=col)
            ax6.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(fig6)

    # ==============================
    # TAB 4: ANALISIS KORELASI
    # ==============================
    with tab4:
        st.subheader("Korelasi Meteorologi vs PM2.5")
        sample_df = filtered_df.sample(min(len(filtered_df), 1600), random_state=42)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig7, ax7 = plt.subplots()
            sns.scatterplot(data=sample_df, x='RAIN', y='PM2.5', alpha=0.5, ax=ax7)
            st.pyplot(fig7)
        with col_c2:
            fig8, ax8 = plt.subplots()
            sns.scatterplot(data=sample_df, x='WSPM', y='PM2.5', alpha=0.5, ax=ax8, color="#0099ff")
            st.pyplot(fig8)

        st.divider()
        st.subheader("Pengaruh Suhu terhadap O3 & SO2")
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig9, ax9 = plt.subplots()
            ax9.scatter(sample_df['TEMP'], sample_df['O3'], alpha=0.3, color="#1900ff")
            ax9.set_title("Suhu vs O3")
            st.pyplot(fig9)
        with col_c4:
            fig10, ax10 = plt.subplots()
            ax10.scatter(sample_df['TEMP'], sample_df['SO2'], alpha=0.3, color="#007bff")
            ax10.set_title("Suhu vs SO2")
            st.pyplot(fig10)

    # ==============================
    # TAB 5: FLUKTUASI HARIAN
    # ==============================
    with tab5:
        st.subheader("Pola Fluktuasi Harian (CO vs NO2)")
        daily_prep = filtered_df.copy()
        daily_prep['hour'] = daily_prep['datetime'].dt.hour
        hourly_mean = daily_prep.groupby('hour')[['CO', 'NO2']].mean()
        
        hourly_std = pd.DataFrame(scaler.fit_transform(hourly_mean), index=hourly_mean.index, columns=hourly_mean.columns)
        
        fig11, ax11 = plt.subplots(figsize=(10, 5))
        ax11.plot(hourly_std.index, hourly_std['CO'], marker='o', label='CO', color="#ff8c00")
        ax11.plot(hourly_std.index, hourly_std['NO2'], marker='o', label='NO2', color="#1900ff")
        ax11.set_xticks(range(0, 24))
        ax11.legend()
        st.pyplot(fig11)

# --- 4️⃣ PAGE: Analisis Lanjutan ---
elif page == "Analisis Lanjutan":
    st.title("💡 Analisis Lanjutan")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Top 5 Konsentrasi Polusi", 
        "Polutan Level", 
        "Korelasi Antar Variabel", 
        "Visualisasi Intensitas PM2.5 Per Stasiun",
        "Visualisasi Intensitas PM10 Per Stasiun"
    ])

    with tab1:
        st.subheader("Top 5 Konsentrasi Polusi")
        top_5_pm25 = cleaned_df.sort_values("PM2.5", ascending=False).head(5)
        top_5_pm10 = cleaned_df.sort_values("PM10", ascending=False).head(5)
        st.dataframe(top_5_pm25, use_container_width=True)
        st.dataframe(top_5_pm10, use_container_width=True)
        st.info(f"Konsentrasi PM2.5 tertinggi tercatat sebesar {top_5_pm25['PM2.5'].iloc[0]} di stasiun {top_5_pm25['station'].iloc[0]} pada {top_5_pm25['datetime'].iloc[0]}")
        st.info(f"Konsentrasi PM10 tertinggi tercatat sebesar {top_5_pm10['PM10'].iloc[0]} di stasiun {top_5_pm10['station'].iloc[0]} pada {top_5_pm10['datetime'].iloc[0]}")

    with tab2:
        st.subheader("Polutan Level")
        treshold_df["station"] = cleaned_df["station"]
        treshold_df["wd"] = cleaned_df["wd"]
        st.dataframe(treshold_df.head())

        level_columns = [col for col in treshold_df.columns if col.endswith("_level")]

        level_summary_df = pd.DataFrame({
            col.replace("_level", ""): treshold_df[col].value_counts()
            for col in level_columns
        }).fillna(0).astype(int)

        st.subheader("Count Polutan Level")
        st.dataframe(level_summary_df)

        colors = ["#15ff00",
                "#f1c40f",
                "#e67e22",
                "#e74c3c",
                "#8b0000"]
        
        distribution_pm25 = (
            treshold_df.groupby(["station", "PM2.5_level"])
            .size()
            .unstack(fill_value=0)
        )

        level_order = ["Sehat", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
        distribution_pm25 = distribution_pm25.reindex(columns=level_order, fill_value=0)

        distribution_pm25.plot(
            kind="bar",
            stacked=True,
            color=colors
        )

        plt.title("Distribusi Level Kualitas Udara PM2.5 per Stasiun")
        plt.xlabel("Stasiun")
        plt.ylabel("Jumlah Observasi")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf())

        distribution_pm10 = (
            treshold_df.groupby(["station", "PM10_level"])
            .size()
            .unstack(fill_value=0)
        )

        level_order = ["Sehat", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
        distribution_pm10 = distribution_pm10.reindex(columns=level_order, fill_value=0)

        distribution_pm10.plot(
            kind="bar",
            stacked=True,
            color=colors
        )

        plt.title("Distribusi Level Kualitas Udara PM10 per Stasiun")
        plt.xlabel("Stasiun")
        plt.ylabel("Jumlah Observasi")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    with tab3:
        st.subheader("Korelasi Antar Variabel")
        num_cols = [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
        ]

        corr_matrix = cleaned_df[num_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5
        )

        plt.title("Heatmap Korelasi Antar Variabel Kualitas Udara dan Cuaca", fontsize=14)
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    with tab4:
        st.subheader("Peta Distribusi PM2.5 per Stasiun")
        station_coords = pd.DataFrame({
            "station": [
                "Dongsi", "Wanliu", "Dingling", "Guanyuan", "Tiantan", "Changping",
                "Aotizhongxin", "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Wanshouxigong"
            ],
            "lat": [
                39.929, 39.992, 40.292, 39.942, 39.886, 40.218,
                39.982, 39.914, 40.328, 39.937, 40.127, 39.878
            ],
            "lon": [
                116.417, 116.287, 116.220, 116.361, 116.407, 116.233,
                116.397, 116.146, 116.628, 116.461, 116.655, 116.352
            ]
        })

        df_geo = cleaned_df.merge(station_coords, on="station", how="left")

        df_summary = (
            df_geo.groupby("station")["PM2.5"]
            .mean()
            .reset_index()
            .merge(station_coords)
        )

        def get_color(pm_value):
            if pm_value < 70:
                return 'green'
            elif pm_value < 80:
                return 'orange'
            else:
                return 'red'

        m = folium.Map(location=[40.0, 116.4], zoom_start=9)

        for _, row in df_summary.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=10,
                popup=f'<b>{row["station"]}</b><br>Rata-rata PM2.5: {row["PM2.5"]:.1f}',
                color=get_color(row["PM2.5"]),
                fill=True,
                fill_color=get_color(row["PM2.5"]),
                fill_opacity=0.7
            ).add_to(m)

        heat_data = df_geo[["lat","lon","PM2.5"]].dropna().sample(n=50000)
        HeatMap(heat_data.values).add_to(m)

        st_folium(m, width=700, height=500)

    with tab5:
        st.subheader("Peta Distribusi PM10 per Stasiun")
        station_coords = pd.DataFrame({
            "station": [
                "Dongsi", "Wanliu", "Dingling", "Guanyuan", "Tiantan", "Changping",
                "Aotizhongxin", "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Wanshouxigong"
            ],
            "lat": [
                39.929, 39.992, 40.292, 39.942, 39.886, 40.218,
                39.982, 39.914, 40.328, 39.937, 40.127, 39.878
            ],
            "lon": [
                116.417, 116.287, 116.220, 116.361, 116.407, 116.233,
                116.397, 116.146, 116.628, 116.461, 116.655, 116.352
            ]
        })

        df_geo = cleaned_df.merge(station_coords, on="station", how="left")

        df_summary = (
            df_geo.groupby("station")["PM10"]
            .mean()
            .reset_index()
            .merge(station_coords)
        )

        def get_color(pm_value):
            if pm_value < 70:
                return 'green'
            elif pm_value < 80:
                return 'orange'
            else:
                return 'red'

        m = folium.Map(location=[40.0, 116.4], zoom_start=9)

        for _, row in df_summary.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=10,
                popup=f'<b>{row["station"]}</b><br>Rata-rata PM10: {row["PM10"]:.1f}',
                color=get_color(row["PM10"]),
                fill=True,
                fill_color=get_color(row["PM10"]),
                fill_opacity=0.7
            ).add_to(m)

        heat_data = df_geo[["lat","lon","PM10"]].dropna().sample(n=50000)
        HeatMap(heat_data.values).add_to(m)
        st_folium(m, width=700, height=500)