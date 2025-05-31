# src/main.py
"""
Integrated Analytics Dashboard
- Fleet TCO Calculator
- Global TBR Market Dashboard
- EV Tire Insight Analytics

"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import logging
import re
import traceback
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# External libraries (install required)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Reddit API library
try:
    import praw
    from tqdm import tqdm
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================
class Config:
    """Application configuration constants"""
    APP_TITLE = "🚀 Integrated Analytics Dashboard"
    VERSION = "1.0.0"

    # Data folder paths
    DATA_FOLDER = "data"
    FLEET_DATA_FILES = {
        "fuel": "fuel_prices.csv",
        "distance": "vehicle_distance.csv",
        "efficiency": "vehicle_efficiency.csv"
    }
    TBR_DATA_FILES = {
        "csv": "trade_data.csv",
        "excel": "tbr_market_data.xlsx",
        "sqlite": "tbr_market.db"
    }
    EV_DATA_FILES = {
        "reddit": "ev_tire_reddit_filtered.csv"
    }

    # Fleet TCO Config
    DEFAULT_TIRE_COUNT = 10
    DEFAULT_TIRE_COST = 250000
    DEFAULT_REPLACE_INTERVAL = 12000

    # TBR Market Config
    TOP_MARKETS_COUNT = 5
    MIN_YEAR = 1990
    MAX_YEAR = 2030

    # EV Tire Config
    EV_KEYWORDS = ['tire', 'tyre', 'regenerative braking', 'noise', 'wear']
    SENTIMENT_THRESHOLD_POS = 0.05
    SENTIMENT_THRESHOLD_NEG = -0.05

# ========================================
# UTILITY FUNCTIONS
# ========================================
def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def format_currency(value: float, currency: str = "KRW") -> str:
    """Format currency with appropriate scaling"""
    if pd.isna(value) or value == 0:
        return f"0 {currency}"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if currency == "USD":
        if abs_value >= 1e12:
            return f"{sign}${abs_value/1e12:,.1f}T"
        elif abs_value >= 1e9:
            return f"{sign}${abs_value/1e9:,.1f}B"
        elif abs_value >= 1e6:
            return f"{sign}${abs_value/1e6:,.1f}M"
        elif abs_value >= 1e3:
            return f"{sign}${abs_value/1e3:,.1f}K"
        else:
            return f"{sign}${abs_value:,.0f}"
    else:  # KRW
        if abs_value >= 1e8:
            return f"{sign}{abs_value/1e8:,.1f}억원"
        elif abs_value >= 1e4:
            return f"{sign}{abs_value/1e4:,.0f}만원"
        else:
            return f"{sign}{abs_value:,.0f}원"

def ensure_data_folder():
    """Ensure data folder exists"""
    if not os.path.exists(Config.DATA_FOLDER):
        os.makedirs(Config.DATA_FOLDER)
        st.info(f"📁 데이터 폴더가 생성되었습니다: {Config.DATA_FOLDER}")

def get_file_path(filename: str) -> str:
    """Get full file path in data folder"""
    return os.path.join(Config.DATA_FOLDER, filename)

def check_file_exists(filename: str) -> bool:
    """Check if file exists in data folder"""
    return os.path.exists(get_file_path(filename))

# ========================================
# DATA PROCESSING FUNCTIONS
# ========================================
class DataProcessor:
    """Common data processing utilities"""

    @staticmethod
    def load_csv_safe(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Safely load CSV file with fallback encodings"""
        encodings = [encoding, 'utf-8-sig', 'cp949', 'euc-kr']

        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"CSV 로드 오류 ({enc}): {str(e)}")
                break

        st.error("지원되는 인코딩으로 파일을 읽을 수 없습니다.")
        return pd.DataFrame()

    @staticmethod
    def load_csv_from_upload(file, encoding: str = 'utf-8') -> pd.DataFrame:
        """Safely load CSV file from upload with fallback encodings"""
        encodings = [encoding, 'utf-8-sig', 'cp949', 'euc-kr']

        for enc in encodings:
            try:
                file.seek(0)  # Reset file pointer
                return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"CSV 로드 오류 ({enc}): {str(e)}")
                break

        st.error("지원되는 인코딩으로 파일을 읽을 수 없습니다.")
        return pd.DataFrame()

    @staticmethod
    def load_excel_safe(file_path: str) -> pd.DataFrame:
        """Safely load Excel file"""
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            st.error(f"Excel 파일 로드 오류: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def load_excel_from_upload(file) -> pd.DataFrame:
        """Safely load Excel file from upload"""
        try:
            return pd.read_excel(file)
        except Exception as e:
            st.error(f"Excel 파일 로드 오류: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate if DataFrame has required columns"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"필수 컬럼이 누락되었습니다: {missing_cols}")
            st.info(f"사용 가능한 컬럼: {list(df.columns)}")
            return False
        return True

# ========================================
# FLEET TCO CALCULATOR
# ========================================
class FleetTCOCalculator:
    """Fleet Total Cost of Ownership Calculator"""

    def __init__(self):
        self.logger = setup_logging()
        self.use_default_data = True

    def load_distance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and process distance data"""
        try:
            # Handle various column name patterns
            distance_col = None
            for col in df.columns:
                if '주행거리' in col or 'distance' in col.lower():
                    distance_col = col
                    break

            if not distance_col:
                st.error("주행거리 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()

            df_clean = df.copy()
            df_clean[distance_col] = pd.to_numeric(
                df_clean[distance_col].astype(str).str.strip(),
                errors='coerce'
            )

            # Standardize column names
            col_mapping = {}
            for col in df_clean.columns:
                if '차종' in col:
                    col_mapping[col] = '차종'
                elif '연료' in col or '유형' in col:
                    col_mapping[col] = '유형'
                elif distance_col == col:
                    col_mapping[col] = '평균주행거리_km'

            df_clean = df_clean.rename(columns=col_mapping)
            return df_clean[['차종', '유형', '평균주행거리_km']].dropna()

        except Exception as e:
            st.error(f"주행거리 데이터 처리 오류: {str(e)}")
            return pd.DataFrame()

    def load_efficiency_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and process fuel efficiency data"""
        try:
            # Find efficiency column
            efficiency_col = None
            for col in df.columns:
                if '연비' in col or 'efficiency' in col.lower():
                    efficiency_col = col
                    break

            if not efficiency_col:
                st.error("연비 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()

            df_clean = df.copy()
            df_clean[efficiency_col] = pd.to_numeric(
                df_clean[efficiency_col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )

            # Find vehicle type column
            vehicle_col = None
            for col in df_clean.columns:
                if '차종' in col:
                    vehicle_col = col
                    break

            if not vehicle_col:
                st.error("차종 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()

            return (
                df_clean[[vehicle_col, efficiency_col]]
                .rename(columns={vehicle_col: '차종', efficiency_col: '복합_연비'})
                .groupby('차종', as_index=False)
                .mean()
            )

        except Exception as e:
            st.error(f"연비 데이터 처리 오류: {str(e)}")
            return pd.DataFrame()

    def load_fuel_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and process fuel price data"""
        try:
            # Find date column
            date_col = None
            for col in df.columns:
                if '구분' in col or 'date' in col.lower() or '날짜' in col:
                    date_col = col
                    break

            if not date_col:
                st.error("날짜 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()

            # Find fuel price column
            fuel_col = None
            for col in df.columns:
                if '경유' in col or 'diesel' in col.lower():
                    fuel_col = col
                    break

            if not fuel_col:
                st.error("경유 가격 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()

            df_clean = df.copy()

            # Parse date - try multiple formats
            try:
                df_clean['date'] = pd.to_datetime(df_clean[date_col], format='%Y년%m월%d일')
            except:
                try:
                    df_clean['date'] = pd.to_datetime(df_clean[date_col])
                except:
                    st.error("날짜 형식을 인식할 수 없습니다.")
                    return pd.DataFrame()

            # Clean price data
            df_clean[fuel_col] = pd.to_numeric(df_clean[fuel_col], errors='coerce')

            return df_clean[['date', fuel_col]].rename(columns={fuel_col: 'ℓ당_가격_원'}).dropna()

        except Exception as e:
            st.error(f"연료 가격 데이터 처리 오류: {str(e)}")
            return pd.DataFrame()


    def reset_state(self):
        """상태 초기화"""
        self.use_default_data = False
        # 세션 상태 초기화
        if 'fuel_df' in st.session_state:
            del st.session_state['fuel_df']
        if 'dist_df' in st.session_state:
            del st.session_state['dist_df']
        if 'eff_df' in st.session_state:
            del st.session_state['eff_df']

    def render(self):
        """Render Fleet TCO Calculator UI"""
        st.header("🚛 Fleet TCO Calculator")
        st.markdown("차량 운영 총비용을 계산하고 분석합니다.")

        # TCO 계산 설정을 상단에 배치
        with st.expander("⚙️ TCO 계산 설정", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                tire_count = st.number_input("타이어 수", min_value=1, max_value=50, value=Config.DEFAULT_TIRE_COUNT)
            with col2:
                cost_per_tire = st.number_input("타이어 단가(원)", min_value=0, value=Config.DEFAULT_TIRE_COST, step=10000)
            with col3:
                replace_interval = st.number_input("교체 주기(km)", min_value=1000, value=Config.DEFAULT_REPLACE_INTERVAL, step=1000)

        # Check if default data files exist
        fuel_file_exists = check_file_exists(Config.FLEET_DATA_FILES["fuel"])
        dist_file_exists = check_file_exists(Config.FLEET_DATA_FILES["distance"])
        eff_file_exists = check_file_exists(Config.FLEET_DATA_FILES["efficiency"])

        all_files_exist = fuel_file_exists and dist_file_exists and eff_file_exists

        # Data source selection
        col1, col2 = st.columns([3, 1])
        with col1:
            if all_files_exist:
                st.success("✅ 기본 데이터 파일이 준비되어 있습니다.")
                self.use_default_data = True
            else:
                st.info("📁 기본 데이터 파일이 없습니다. 파일을 업로드하세요.")
                self.use_default_data = False

        with col2:
            if st.button("🔄 초기화", key="fleet_reset", type="secondary"):
                self.reset_state()
                st.success("✅ 데이터가 초기화되었습니다!")
                st.rerun()

        # File uploads or use default data
        fuel_df = pd.DataFrame()
        dist_df = pd.DataFrame()
        eff_df = pd.DataFrame()

        if self.use_default_data and all_files_exist:
            # Load default data
            fuel_df = DataProcessor.load_csv_safe(get_file_path(Config.FLEET_DATA_FILES["fuel"]))
            dist_df = DataProcessor.load_csv_safe(get_file_path(Config.FLEET_DATA_FILES["distance"]))
            eff_df = DataProcessor.load_csv_safe(get_file_path(Config.FLEET_DATA_FILES["efficiency"]))

            st.info(f"📊 기본 데이터 사용 중: {Config.FLEET_DATA_FILES}")

        else:
            # File upload interface
            st.subheader("📁 데이터 파일 업로드")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**주유소 평균 가격 데이터**")
                st.caption("예: fuel_prices.csv")
                fuel_file = st.file_uploader("파일 선택", type=['csv'], key='fuel_file')
                if fuel_file:
                    fuel_df = DataProcessor.load_csv_from_upload(fuel_file)

            with col2:
                st.markdown("**차량 주행거리 데이터**")
                st.caption("예: vehicle_distance.csv")
                dist_file = st.file_uploader("파일 선택", type=['csv'], key='dist_file')
                if dist_file:
                    dist_df = DataProcessor.load_csv_from_upload(dist_file)

            with col3:
                st.markdown("**차량 연비 데이터**")
                st.caption("예: vehicle_efficiency.csv")
                eff_file = st.file_uploader("파일 선택", type=['csv'], key='eff_file')
                if eff_file:
                    eff_df = DataProcessor.load_csv_from_upload(eff_file)

        # TCO 계산 및 결과 표시
        if not fuel_df.empty and not dist_df.empty and not eff_df.empty:
            try:
                # Process data
                dist_data = self.load_distance_data(dist_df)
                eff_data = self.load_efficiency_data(eff_df)
                fuel_data = self.load_fuel_price_data(fuel_df)

                if dist_data.empty or eff_data.empty or fuel_data.empty:
                    st.error("데이터 처리에 실패했습니다.")
                    return

                # Merge and calculate
                merged_data = pd.merge(dist_data, eff_data, on='차종', how='inner')

                # Filter for trucks and diesel - .loc 사용으로 수정
                mask = (merged_data['차종'].str.contains('화물', na=False)) & \
                       (merged_data['유형'].str.contains('경유', na=False))

                truck_data = merged_data.loc[mask].copy()  # 명시적으로 .copy() 사용

                if truck_data.empty:
                    st.warning("화물차 + 경유 조합 데이터가 없습니다. 전체 데이터로 계산합니다.")
                    truck_data = merged_data.iloc[:1].copy()  # .iloc 사용

                # .loc을 사용해서 새 컬럼 추가
                truck_data.loc[:, '일일연료소비량_ℓ'] = truck_data['평균주행거리_km'] / truck_data['복합_연비']

                # Expand data with fuel prices
                truck_data.loc[:, 'key'] = 1
                fuel_data_copy = fuel_data.copy()
                fuel_data_copy.loc[:, 'key'] = 1

                expanded_data = pd.merge(truck_data, fuel_data_copy, on='key').drop('key', axis=1)

                # Calculate costs - .loc 사용
                expanded_data.loc[:, '일일연료비_원'] = \
                    expanded_data['일일연료소비량_ℓ'] * expanded_data['ℓ당_가격_원']

                expanded_data.loc[:, 'year_month'] = expanded_data['date'].dt.to_period('M')


                monthly_cost = (
                    expanded_data
                    .groupby('year_month', as_index=False)['일일연료비_원']
                    .sum()
                    .rename(columns={'일일연료비_원': '월간연료비_원'})
                )

                # Calculate annual TCO
                avg_km = truck_data['평균주행거리_km'].iloc[0]
                annual_km = avg_km * 365
                annual_fuel = monthly_cost['월간연료비_원'].sum()
                annual_tire = (annual_km / replace_interval) * tire_count * cost_per_tire
                annual_tco = annual_fuel + annual_tire

                # Display results
                st.subheader("📊 연간 비용 요약")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("연간 연료비", format_currency(annual_fuel))
                with col2:
                    st.metric("연간 타이어비용", format_currency(annual_tire))
                with col3:
                    st.metric("연간 TCO", format_currency(annual_tco))

                # Charts
                st.subheader("📈 월간 연료비 추이")
                monthly_chart_data = monthly_cost.copy()
                monthly_chart_data['date'] = monthly_chart_data['year_month'].dt.to_timestamp()

                fig = px.line(
                    monthly_chart_data,
                    x='date',
                    y='월간연료비_원',
                    title="월간 연료비 추이"
                )
                fig.update_layout(xaxis_title="날짜", yaxis_title="연료비 (원)")
                st.plotly_chart(fig, use_container_width=True)

                # Fuel price trend
                st.subheader("⛽ 연료 가격 추이")
                fig2 = px.line(
                    fuel_data,
                    x='date',
                    y='ℓ당_가격_원',
                    title="연료 가격 추이"
                )
                fig2.update_layout(xaxis_title="날짜", yaxis_title="가격 (원/ℓ)")
                st.plotly_chart(fig2, use_container_width=True)

                # Cost breakdown pie chart
                st.subheader("💰 연간 비용 구성")
                cost_data = pd.DataFrame({
                    'Category': ['연료비', '타이어비용'],
                    'Amount': [annual_fuel, annual_tire]
                })

                fig3 = px.pie(
                    cost_data,
                    values='Amount',
                    names='Category',
                    title="연간 비용 구성비"
                )
                st.plotly_chart(fig3, use_container_width=True)

            except Exception as e:
                st.error(f"TCO 계산 중 오류가 발생했습니다: {str(e)}")
                st.text(traceback.format_exc())
        else:
            if not self.use_default_data:
                st.info("📁 CSV 파일 3개를 모두 업로드해주세요.")

# ========================================
# TBR MARKET DASHBOARD
# ========================================
# src/tbr_market_dashboard.py (수정된 부분)

class TBRMarketDashboard:
    """TBR Market Analysis Dashboard"""

    def __init__(self):
        self.logger = setup_logging()
        self.use_default_data = True
        self.column_mappings = {
            "country": ["reporteriso", "reporterISO", "reporter_iso", "country", "reporter"],
            "year": ["refmonth", "refMonth", "ref_month", "period", "year"],
            "flow": ["flowcode", "flowCode", "flow_code", "flowdesc", "flow"],
            "value": ["cifvalue", "fobvalue", "primaryvalue", "tradevalue", "value"]
        }
        self.flow_mappings = {
            'M': 'Import', 'X': 'Export', 1: 'Import', 2: 'Export',
            '1': 'Import', '2': 'Export', 'I': 'Import', 'E': 'Export'
        }

    def get_database_info(self, db_path: str) -> Dict[str, Any]:
        """Get database structure information with detailed debugging"""
        try:
            # Check if file exists
            if not os.path.exists(db_path):
                st.error(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
                return {'tables': [], 'table_info': {}}

            # Check file size
            file_size = os.path.getsize(db_path)

            if file_size == 0:
                st.error("데이터베이스 파일이 비어있습니다.")
                return {'tables': [], 'table_info': {}}

            with sqlite3.connect(db_path) as conn:
                # Get all tables
                tables_query = "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name;"

                try:
                    tables_df = pd.read_sql(tables_query, conn)
                    tables = tables_df['name'].tolist()
                except Exception as e:
                    # Try alternative approach
                    try:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        results = cursor.fetchall()
                        tables = [row[0] for row in results]
                    except Exception as e2:
                        return {'tables': [], 'table_info': {}}

                if not tables:
                    return {'tables': [], 'table_info': {}}

                # Get columns for each table
                table_info = {}
                for table in tables:
                    try:
                        columns_query = f"PRAGMA table_info([{table}]);"
                        columns_df = pd.read_sql(columns_query, conn)

                        count_query = f"SELECT COUNT(*) as count FROM [{table}];"
                        count_result = pd.read_sql(count_query, conn)

                        table_info[table] = {
                            'columns': columns_df['name'].tolist(),
                            'row_count': count_result['count'].iloc[0]
                        }

                    except Exception as e:
                        table_info[table] = {
                            'columns': [],
                            'row_count': 0
                        }

                return {
                    'tables': tables,
                    'table_info': table_info
                }

        except sqlite3.DatabaseError as e:
            st.error(f"SQLite 데이터베이스 오류: {str(e)}")
            return {'tables': [], 'table_info': {}}
        except Exception as e:
            st.error(f"데이터베이스 정보 조회 오류: {str(e)}")
            return {'tables': [], 'table_info': {}}

    def auto_load_best_table(self, db_path: str) -> Tuple[pd.DataFrame, str]:
        """자동으로 가장 적합한 테이블을 찾아서 로드"""
        try:
            with sqlite3.connect(db_path) as conn:
                # Get database info
                db_info = self.get_database_info(db_path)

                if not db_info['tables']:
                    return pd.DataFrame(), ""

                # 테이블 우선순위 결정
                best_table = None
                max_score = 0

                for table_name, info in db_info['table_info'].items():
                    score = 0

                    # 1. 데이터 행 수 (많을수록 좋음)
                    score += min(info['row_count'] / 10000, 5)  # 최대 5점

                    # 2. 테이블 이름 점수 (trade 관련 키워드)
                    name_keywords = ['trade', 'data', 'main', 'market', 'export', 'import']
                    for keyword in name_keywords:
                        if keyword in table_name.lower():
                            score += 3

                    # 3. 컬럼 매칭 점수 (필요한 컬럼들이 있는지)
                    required_columns = ['country', 'year', 'flow', 'value']
                    column_matches = 0

                    for req_col, candidates in self.column_mappings.items():
                        for candidate in candidates:
                            if candidate in [col.lower() for col in info['columns']]:
                                column_matches += 1
                                break

                    score += column_matches * 2  # 컬럼 매치당 2점

                    if score > max_score:
                        max_score = score
                        best_table = table_name

                if best_table:
                    # 선택된 테이블 로드
                    query = f"SELECT * FROM [{best_table}]"
                    df = pd.read_sql(query, conn)

                    return df, best_table

                return pd.DataFrame(), ""

        except Exception as e:
            st.error(f"자동 테이블 로드 오류: {str(e)}")
            return pd.DataFrame(), ""

    def load_data_from_database(self, db_path: str, auto_load: bool = True) -> pd.DataFrame:
        """Load data from database with optional auto-loading"""
        try:
            with sqlite3.connect(db_path) as conn:
                # Get database info
                db_info = self.get_database_info(db_path)

                if not db_info['tables']:
                    st.error("데이터베이스에 테이블이 없습니다.")
                    return pd.DataFrame()

                # 자동 로드 모드
                if auto_load:
                    df_auto, best_table = self.auto_load_best_table(db_path)

                    if not df_auto.empty:
                        st.success(f"✅ 자동으로 최적 테이블 '{best_table}' 로드 완료 ({len(df_auto):,}행)")

                        # 데이터베이스 구조 정보 (접을 수 있는 형태로)
                        with st.expander("🗄️ 데이터베이스 구조 정보"):
                            for table_name, info in db_info['table_info'].items():
                                selected_indicator = "🎯 " if table_name == best_table else "📋 "
                                st.markdown(f"**{selected_indicator}{table_name}** ({info['row_count']:,}행)")
                                st.caption(f"컬럼: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")

                        # 데이터 미리보기
                        with st.expander("📋 데이터 미리보기", expanded=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.dataframe(df_auto.head(10))
                            with col2:
                                st.metric("총 행 수", f"{len(df_auto):,}")
                                st.metric("총 컬럼 수", len(df_auto.columns))

                        return df_auto

                # 수동 선택 모드 (자동 로드 실패 시 또는 사용자가 다른 테이블 선택 시)
                st.subheader("🗄️ 테이블 수동 선택")

                for table_name, info in db_info['table_info'].items():
                    with st.expander(f"📋 테이블: {table_name} ({info['row_count']:,}행)"):
                        st.write("**컬럼:**", ", ".join(info['columns']))

                # 테이블 선택
                common_names = ['trade_data', 'trades', 'data', 'main', 'export_data', 'import_data']
                selected_table = None

                for common_name in common_names:
                    if common_name in db_info['tables']:
                        selected_table = common_name
                        break

                if selected_table is None:
                    selected_table = db_info['tables'][0]

                selected_table = st.selectbox(
                    "분석할 테이블 선택",
                    options=db_info['tables'],
                    index=db_info['tables'].index(selected_table) if selected_table in db_info['tables'] else 0
                )

                if st.button("선택된 테이블 로드"):
                    query = f"SELECT * FROM [{selected_table}]"
                    df = pd.read_sql(query, conn)
                    st.success(f"✅ 테이블 '{selected_table}'에서 {len(df):,}행 로드")

                    with st.expander("📋 데이터 미리보기"):
                        st.dataframe(df.head(10))

                    return df

                return pd.DataFrame()

        except Exception as e:
            st.error(f"데이터베이스 로드 오류: {str(e)}")
            return pd.DataFrame()

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Intelligent column detection"""
        column_map = {}

        for target_col, candidates in self.column_mappings.items():
            for candidate in candidates:
                if candidate in df.columns:
                    column_map[target_col] = candidate
                    break

        return column_map

    def process_trade_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process trade data with intelligent column mapping"""
        try:
            df_work = df.copy()
            df_work.columns = [c.lower().strip() for c in df_work.columns]

            column_map = self.detect_columns(df_work)

            if len(column_map) < 4:
                missing = set(['country', 'year', 'flow', 'value']) - set(column_map.keys())
                st.error(f"필수 컬럼 매핑 실패: {missing}")
                st.info(f"사용 가능한 컬럼: {list(df_work.columns)}")
                return pd.DataFrame()

            # Apply mapping
            rename_dict = {v: k for k, v in column_map.items()}
            df_clean = df_work.rename(columns=rename_dict)[['country', 'year', 'flow', 'value']].copy()

            # Clean data
            df_clean['country'] = df_clean['country'].astype(str).str.strip().str.upper()
            df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')

            # Extract year from various formats
            def extract_year(val):
                try:
                    val_str = str(val)
                    if len(val_str) == 6 and val_str.isdigit():
                        return int(val_str[:4])
                    return int(float(val))
                except:
                    return None

            df_clean['year'] = df_clean['year'].apply(extract_year)

            # Map flow values
            def map_flow(val):
                if val in self.flow_mappings:
                    return self.flow_mappings[val]
                return str(val).title()

            df_clean['flow'] = df_clean['flow'].apply(map_flow)

            # Filter valid data
            df_clean = df_clean.dropna(subset=['country', 'year', 'flow', 'value'])
            df_clean = df_clean[
                (df_clean['year'] >= Config.MIN_YEAR) &
                (df_clean['year'] <= Config.MAX_YEAR) &
                (df_clean['value'] >= 0)
                ]

            return df_clean

        except Exception as e:
            st.error(f"데이터 처리 오류: {str(e)}")
            return pd.DataFrame()

    def render_tableau_dashboard(self):
        """Render Tableau dashboard when using SQLite"""
        st.subheader("📊 Tableau 대시보드")
        st.markdown("SQLite 데이터베이스 연결 시 Tableau 시각화를 제공합니다.")

        tableau_html = """
       <div class='tableauPlaceholder' id='viz1748617557024' style='position: relative'>
           <noscript>
               <a href='#'>
                   <img alt='대시보드 1 ' src='https://public.tableau.com/static/images/Gl/GlobalTBRMarketDashboard/1/1_rss.png' style='border: none' />
               </a>
           </noscript>
           <object class='tableauViz' style='display:none;'>
               <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
               <param name='embed_code_version' value='3' />
               <param name='site_root' value='' />
               <param name='name' value='GlobalTBRMarketDashboard&#47;1' />
               <param name='tabs' value='no' />
               <param name='toolbar' value='yes' />
               <param name='static_image' value='https://public.tableau.com/static/images/Gl/GlobalTBRMarketDashboard/1/1.png' />
               <param name='animate_transition' value='yes' />
               <param name='display_static_image' value='yes' />
               <param name='display_spinner' value='yes' />
               <param name='display_overlay' value='yes' />
               <param name='display_count' value='yes' />
               <param name='language' value='ko-KR' />
               <param name='filter' value='publish=yes' />
           </object>
       </div>
       <script type='text/javascript'>
           var divElement = document.getElementById('viz1748617557024');
           var vizElement = divElement.getElementsByTagName('object')[0];
           if (divElement.offsetWidth > 800) {
               vizElement.style.minWidth='1000px';
               vizElement.style.maxWidth='100%';
               vizElement.style.minHeight='1500px';
               vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';
           } else if (divElement.offsetWidth > 500) {
               vizElement.style.minWidth='420px';
               vizElement.style.maxWidth='100%';
               vizElement.style.minHeight='827px';
               vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';
           } else {
               vizElement.style.width='100%';
               vizElement.style.height='927px';
           }
           var scriptElement = document.createElement('script');
           scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
           vizElement.parentNode.insertBefore(scriptElement, vizElement);
       </script>
       """

        st.components.v1.html(tableau_html, height=1600)

    def reset_state(self):
        """상태 초기화"""
        self.use_default_data = False
        if 'tbr_df' in st.session_state:
            del st.session_state['tbr_df']

    def render(self):
        """Render TBR Market Dashboard UI (수정된 버전)"""
        st.header("🌐 Global TBR Market Dashboard")
        st.markdown("글로벌 TBR 시장 데이터를 분석합니다.")

        # Check if default data files exist
        sqlite_file_exists = check_file_exists(Config.TBR_DATA_FILES["sqlite"])
        csv_file_exists = check_file_exists(Config.TBR_DATA_FILES["csv"])
        excel_file_exists = check_file_exists(Config.TBR_DATA_FILES["excel"])

        any_file_exists = sqlite_file_exists or csv_file_exists or excel_file_exists

        # Data source selection
        col1, col2 = st.columns([3, 1])
        with col1:
            if any_file_exists:
                available_files = []
                if sqlite_file_exists:
                    available_files.append("SQLite DB")
                if csv_file_exists:
                    available_files.append("CSV")
                if excel_file_exists:
                    available_files.append("Excel")

                st.success(f"✅ 기본 데이터 파일 준비됨: {', '.join(available_files)}")
                self.use_default_data = True
            else:
                st.info("📁 기본 데이터 파일이 없습니다. 파일을 업로드하세요.")
                self.use_default_data = False

        with col2:
            if st.button("🔄 초기화", key="tbr_reset"):
                self.use_default_data = False
                st.rerun()

        # 기본적으로 SQLite 데이터베이스 우선 선택
        df_trade = pd.DataFrame()

        # SQLite 파일이 있으면 자동으로 로드 시도
        if sqlite_file_exists:
            st.info("🔄 SQLite 데이터베이스에서 자동 로드 중...")

            try:
                # 자동 로드 시도
                df_raw = self.load_data_from_database(
                    get_file_path(Config.TBR_DATA_FILES["sqlite"]),
                    auto_load=True
                )

                # 1. 자동 로드 성공 시 Tableau 대시보드 표시 부분
                if not df_raw.empty:
                    df_trade = self.process_trade_data(df_raw)
                    if not df_trade.empty:
                        st.success("✅ SQLite 데이터베이스에서 자동 로드 완료!")

                        # Tableau 대시보드 표시 (기본적으로 펼쳐짐)
                        with st.expander("📊 Tableau 대시보드 보기", expanded=True):
                            self.render_tableau_dashboard()
                    else:
                        st.warning("⚠️ 데이터 처리에 실패했습니다. 수동으로 다른 옵션을 선택해주세요.")

            except Exception as e:
                st.error(f"❌ 자동 로드 실패: {str(e)}")
                st.info("💡 수동으로 데이터 소스를 선택해주세요.")

        # 자동 로드가 실패했거나 SQLite 파일이 없는 경우에만 선택 옵션 표시
        if df_trade.empty:
            st.markdown("---")
            st.subheader("📁 데이터 소스 선택")

            if self.use_default_data and any_file_exists:
                data_source = st.radio(
                    "데이터 소스 선택",
                    ["SQLite 데이터베이스", "기본 CSV/Excel", "파일 업로드"],
                    key="tbr_data_source"
                )
            else:
                data_source = st.radio(
                    "데이터 소스 선택",
                    ["SQLite 데이터베이스", "파일 업로드"],
                    key="tbr_data_source_upload"
                )

            if data_source == "SQLite 데이터베이스":
                db_path = st.text_input("데이터베이스 경로", value=get_file_path(Config.TBR_DATA_FILES["sqlite"]))

                if st.button("🔄 수동으로 데이터베이스 로드"):
                    df_raw = self.load_data_from_database(db_path, auto_load=False)
                    if not df_raw.empty:
                        df_trade = self.process_trade_data(df_raw)
                        if not df_trade.empty:
                            st.success("✅ SQLite 데이터베이스 연결 성공")
                            # Tableau 대시보드 표시 (기본적으로 펼쳐짐)
                            with st.expander("📊 Tableau 대시보드 보기", expanded=True):
                                self.render_tableau_dashboard()


            elif data_source == "기본 CSV/Excel" and self.use_default_data:
                if csv_file_exists:
                    df_raw = DataProcessor.load_csv_safe(get_file_path(Config.TBR_DATA_FILES["csv"]))
                    if not df_raw.empty:
                        df_trade = self.process_trade_data(df_raw)
                        if not df_trade.empty:
                            st.success("✅ SQLite 데이터베이스에서 자동 로드 완료!")

                            # 섹션으로 구분하여 표시
                            st.markdown("---")
                            st.subheader("📊 Tableau 대시보드")
                            self.render_tableau_dashboard()

                elif excel_file_exists:
                    df_raw = DataProcessor.load_excel_safe(get_file_path(Config.TBR_DATA_FILES["excel"]))
                    if not df_raw.empty:
                        df_trade = self.process_trade_data(df_raw)
                        st.success("✅ Excel 파일에서 데이터 로드")

            else:  # File upload
                st.markdown("**TBR 거래 데이터 업로드**")
                st.caption("예: trade_data.csv, tbr_market_data.xlsx")
                uploaded_file = st.file_uploader("파일 선택", type=['csv', 'xlsx'], key='tbr_file')

                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df_raw = DataProcessor.load_csv_from_upload(uploaded_file)
                        else:
                            df_raw = DataProcessor.load_excel_from_upload(uploaded_file)

                        if not df_raw.empty:
                            df_trade = self.process_trade_data(df_raw)
                            st.success("✅ 업로드 파일에서 데이터 로드")
                    except Exception as e:
                        st.error(f"파일 처리 오류: {str(e)}")

        # Display analysis results (기존 코드와 동일)
        if not df_trade.empty:
            # Display data info
            st.success(f"✅ 데이터 분석 준비 완료: {len(df_trade):,}행")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 거래건수", f"{len(df_trade):,}")
            with col2:
                st.metric("국가 수", df_trade['country'].nunique())
            with col3:
                st.metric("연도 범위", f"{df_trade['year'].min()}-{df_trade['year'].max()}")
            with col4:
                st.metric("총 거래액", format_currency(df_trade['value'].sum(), "USD"))

            # Interactive filters
            st.subheader("🔍 데이터 필터")

            # TBR 분석 설정을 탭 내부에 배치
            with st.expander("🔧 TBR 분석 설정", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    selected_years = st.multiselect(
                        "연도 선택",
                        options=sorted(df_trade['year'].unique()),
                        default=sorted(df_trade['year'].unique())[-3:] if len(df_trade['year'].unique()) >= 3 else sorted(df_trade['year'].unique())
                    )

                with col2:
                    selected_flows = st.multiselect(
                        "거래 유형",
                        options=df_trade['flow'].unique(),
                        default=df_trade['flow'].unique()
                    )

                with col3:
                    selected_countries = st.multiselect(
                        "국가 선택 (빈 값 = 전체)",
                        options=sorted(df_trade['country'].unique())
                    )

            # Apply filters
            filtered_df = df_trade[
                (df_trade['year'].isin(selected_years)) &
                (df_trade['flow'].isin(selected_flows))
                ]

            if selected_countries:
                filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

            # Visualizations (기존 코드와 동일)
            if not filtered_df.empty:
                # Yearly trend
                st.subheader("📊 연도별 거래량 추이")
                yearly_data = filtered_df.pivot_table(
                    index='year',
                    columns='flow',
                    values='value',
                    aggfunc='sum'
                ).fillna(0)

                fig = px.line(yearly_data, title="연도별 거래량 추이 (USD)")
                fig.update_layout(xaxis_title="연도", yaxis_title="거래액 (USD)")
                st.plotly_chart(fig, use_container_width=True)

                # Top markets
                st.subheader("🏆 주요 시장 분석")
                latest_year = filtered_df['year'].max()

                for flow_type in selected_flows:
                    flow_data = filtered_df[
                        (filtered_df['flow'] == flow_type) &
                        (filtered_df['year'] == latest_year)
                        ]

                    if not flow_data.empty:
                        top_markets = (
                            flow_data
                            .groupby('country')['value']
                            .sum()
                            .sort_values(ascending=False)
                            .head(10)
                        )

                        fig = px.bar(
                            x=top_markets.values,
                            y=top_markets.index,
                            orientation='h',
                            title=f"{flow_type} 상위 10개 시장 ({latest_year}년)"
                        )
                        fig.update_layout(xaxis_title="거래액 (USD)", yaxis_title="국가")
                        st.plotly_chart(fig, use_container_width=True)

                # Data table
                with st.expander("📋 상세 데이터"):
                    st.dataframe(filtered_df.head(100))

            else:
                st.warning("필터 조건에 맞는 데이터가 없습니다.")

        else:
            if not sqlite_file_exists:
                st.info("💡 SQLite 데이터베이스 파일을 data 폴더에 추가하시거나 다른 데이터 소스를 선택해주세요.")
# ========================================
# EV TIRE INSIGHT ANALYTICS
# ========================================
class EVTireInsightAnalytics:
    """EV Tire Insight Analytics from Reddit Data"""

    def __init__(self):
        self.logger = setup_logging()
        self.use_default_data = True
        # 이 3줄을 추가하세요
        self.stop_words = None
        self.lemmatizer = None
        self.sia = None
        self.setup_nltk()

    def setup_nltk(self):
        """Setup NLTK resources with error handling"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)

                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
                self.sia = SentimentIntensityAnalyzer()

            except Exception as e:
                st.warning(f"NLTK 설정 오류: {str(e)}")
                # 실패 시 기본값 설정
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
                self.lemmatizer = None
                self.sia = None
        else:
            # NLTK 없을 때 기본값
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
            self.lemmatizer = None
            self.sia = None

    def collect_reddit_data(self, client_id: str, client_secret: str, user_agent: str,
                            subreddits: List[str], keywords: List[str], limit: int = 500) -> pd.DataFrame:
        """Collect Reddit data using PRAW"""
        if not REDDIT_AVAILABLE:
            st.error("PRAW 라이브러리가 설치되지 않았습니다. `pip install praw tqdm`을 실행하세요.")
            return pd.DataFrame()

        try:
            # Reddit API 인증
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            reddit.read_only = True

            records = []

            # Progress bar setup
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_subreddits = len(subreddits)

            for idx, sub in enumerate(subreddits):
                status_text.text(f"수집 중: r/{sub}")

                try:
                    subreddit = reddit.subreddit(sub)
                    posts = list(subreddit.hot(limit=limit))

                    for post in posts:
                        title = (post.title or "").lower()
                        selftext = (post.selftext or "").lower()

                        # 키워드 매치 개수 세기
                        match_count = sum(
                            (kw.lower() in title) + (kw.lower() in selftext)
                            for kw in keywords
                        )

                        # 두 가지 키워드 이상 포함된 경우만 저장
                        if match_count >= 2:
                            records.append({
                                'id': post.id,
                                'created_utc': datetime.fromtimestamp(post.created_utc),
                                'subreddit': sub,
                                'author': str(post.author),
                                'title': post.title,
                                'selftext': post.selftext,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'url': post.url,
                                'matched_keywords': match_count,
                                'type': 'post'
                            })

                        # 댓글도 수집
                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments.list()[:50]:  # 댓글 수 제한
                                body = (comment.body or "").lower()
                                match_count = sum(kw.lower() in body for kw in keywords)
                                if match_count >= 2:
                                    records.append({
                                        'id': comment.id,
                                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                                        'subreddit': sub,
                                        'parent_id': comment.parent_id,
                                        'author': str(comment.author),
                                        'body': comment.body,
                                        'score': comment.score,
                                        'type': 'comment',
                                        'matched_keywords': match_count
                                    })
                        except Exception as comment_error:
                            st.warning(f"댓글 수집 오류 (r/{sub}): {str(comment_error)}")
                            continue

                except Exception as sub_error:
                    st.error(f"서브레딧 r/{sub} 접근 오류: {str(sub_error)}")
                    continue

                # Progress update
                progress_bar.progress((idx + 1) / total_subreddits)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            if records:
                df = pd.DataFrame(records)
                st.success(f"✅ Reddit 데이터 수집 완료: {len(df)}건")
                return df
            else:
                st.warning("수집된 데이터가 없습니다.")
                return pd.DataFrame()

        except Exception as e:
            st.error(f"Reddit 데이터 수집 오류: {str(e)}")
            return pd.DataFrame()

    def render_reddit_collector(self):
        """Render Reddit data collection interface"""
        st.subheader("🔍 Reddit 데이터 수집")
        st.markdown("Reddit API를 통해 EV 타이어 관련 게시물과 댓글을 수집합니다.")

        if not REDDIT_AVAILABLE:
            st.error("🚨 PRAW 라이브러리가 필요합니다.")
            st.code("pip install praw tqdm")
            return pd.DataFrame()

        # API 설정
        with st.expander("🔑 Reddit API 설정", expanded=True):
            st.markdown("""
           Reddit API 사용을 위해 [Reddit App](https://www.reddit.com/prefs/apps)에서 앱을 생성하고 정보를 입력하세요.
           """)

            col1, col2 = st.columns(2)
            with col1:
                client_id = st.text_input("Client ID", value="2XU62DdrTzSJUt6Wsy7xuA")
                user_agent = st.text_input("User Agent", value="ev_tire_insights/0.1 by your_username")

            with col2:
                client_secret = st.text_input("Client Secret", type="password", value="FPar49NZEEhGP4C4zL9pdeWXjvnGdA")

        # 수집 설정
        with st.expander("⚙️ 수집 설정", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                subreddits_input = st.text_area(
                    "대상 서브레딧 (줄바꿈으로 구분)",
                    value="electricvehicles\ntires\nTesla\nEVs"
                )
                subreddits = [s.strip() for s in subreddits_input.split('\n') if s.strip()]

                limit = st.number_input("서브레딧당 게시물 수", min_value=10, max_value=1000, value=500)

            with col2:
                keywords_input = st.text_area(
                    "검색 키워드 (줄바꿈으로 구분)",
                    value="tire\ntyre\nregenerative braking\nnoise\nwear"
                )
                keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

                min_keywords = st.number_input("최소 키워드 매치 수", min_value=1, max_value=5, value=2)

        # 미리보기
        st.info(f"🎯 대상: {len(subreddits)}개 서브레딧, {len(keywords)}개 키워드")

        # 수집 실행
        collected_data = pd.DataFrame()

        if st.button("🚀 데이터 수집 시작", type="primary"):
            if not client_id or not client_secret or not user_agent:
                st.error("Reddit API 정보를 모두 입력해주세요.")
                return pd.DataFrame()

            if not subreddits or not keywords:
                st.error("서브레딧과 키워드를 입력해주세요.")
                return pd.DataFrame()

            with st.spinner("Reddit 데이터 수집 중..."):
                collected_data = self.collect_reddit_data(
                    client_id, client_secret, user_agent,
                    subreddits, keywords, limit
                )

            if not collected_data.empty:
                # 수집 결과 표시
                st.subheader("📊 수집 결과")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("총 수집 건수", len(collected_data))
                with col2:
                    post_count = len(collected_data[collected_data['type'] == 'post']) if 'type' in collected_data.columns else 0
                    st.metric("게시물", post_count)
                with col3:
                    comment_count = len(collected_data[collected_data['type'] == 'comment']) if 'type' in collected_data.columns else 0
                    st.metric("댓글", comment_count)
                with col4:
                    avg_score = collected_data['score'].mean() if 'score' in collected_data.columns else 0
                    st.metric("평균 점수", f"{avg_score:.1f}")

                # 수집된 데이터 미리보기
                with st.expander("📋 수집된 데이터 미리보기"):
                    st.dataframe(collected_data.head(10))

                # 데이터 저장 옵션
                if st.button("💾 데이터 저장"):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"reddit_collected_{timestamp}.csv"
                    filepath = get_file_path(filename)

                    collected_data.to_csv(filepath, index=False, encoding='utf-8-sig')
                    st.success(f"✅ 데이터가 저장되었습니다: {filepath}")

        return collected_data

    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if not isinstance(text, str):
            return ""

        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"[^\w\s]", " ", text)  # Remove special characters
        text = text.lower().strip()  # Lowercase and trim
        text = re.sub(r"\s+", " ", text)  # Multiple spaces to single
        return text

    def process_reddit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Reddit data for analysis"""
        try:
            df_clean = df.copy()

            # Combine text fields
            df_clean["text_raw"] = df_clean.apply(
                lambda r: r["body"] if pd.notna(r.get("body")) and str(r.get("body")).strip()
                else f"{r.get('title', '') or ''} {r.get('selftext', '') or ''}",
                axis=1
            )

            # Clean text
            df_clean["text_clean"] = df_clean["text_raw"].apply(self.clean_text)

            # Parse datetime
            df_clean['created_utc'] = pd.to_datetime(df_clean['created_utc'])

            return df_clean

        except Exception as e:
            st.error(f"Reddit 데이터 처리 오류: {str(e)}")
            return pd.DataFrame()

    def perform_tfidf_analysis(self, df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
        """Perform TF-IDF analysis"""
        try:
            if not NLTK_AVAILABLE:
                st.warning("NLTK가 설치되지 않아 간단한 키워드 분석만 수행합니다.")
                # Simple word frequency analysis
                all_text = ' '.join(df['text_clean'].fillna(''))
                words = all_text.split()
                word_freq = pd.Series(words).value_counts().head(max_features)
                return pd.DataFrame({
                    'Keyword': word_freq.index,
                    'Frequency': word_freq.values
                })

            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 1)
            )

            tfidf_matrix = vectorizer.fit_transform(df["text_clean"].fillna(''))
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1

            tfidf_df = pd.DataFrame({
                'Keyword': feature_names,
                'TF-IDF_Score': tfidf_scores
            }).sort_values('TF-IDF_Score', ascending=False)

            return tfidf_df

        except Exception as e:
            st.error(f"TF-IDF 분석 오류: {str(e)}")
            return pd.DataFrame()

    def perform_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis"""
        try:

            if self.sia is None:
                if NLTK_AVAILABLE:
                    try:
                        self.sia = SentimentIntensityAnalyzer()
                    except:
                        pass

            if self.sia is None:
                st.warning("NLTK 감성 분석을 사용할 수 없습니다. 간단한 분석을 수행합니다.")
                return self.simple_sentiment_fallback(df)

            if not NLTK_AVAILABLE:
                st.warning("NLTK가 설치되지 않아 감성 분석을 수행할 수 없습니다.")
                return df

            # Calculate sentiment scores
            df['sentiment_score'] = df['text_clean'].apply(
                lambda t: self.sia.polarity_scores(t)['compound']
            )

            # Classify sentiment
            df['sentiment'] = df['sentiment_score'].apply(
                lambda s: 'Positive' if s >= Config.SENTIMENT_THRESHOLD_POS
                else ('Negative' if s <= Config.SENTIMENT_THRESHOLD_NEG else 'Neutral')
            )

            return df

        except Exception as e:
            st.error(f"감성 분석 오류: {str(e)}")
            return df

    def tokenize_and_lemmatize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenize and lemmatize text"""
        try:
            if not NLTK_AVAILABLE:
                # Simple tokenization
                df["tokens"] = df["text_clean"].str.split()
                return df

            # Tokenize
            df["tokens"] = df["text_clean"].str.split()

            # Remove stop words
            df["tokens_nostop"] = df["tokens"].apply(
                lambda toks: [t for t in toks if t not in self.stop_words]
                if isinstance(toks, list) else []
            )

            # Lemmatize
            df["tokens_lemmatized"] = df["tokens_nostop"].apply(
                lambda toks: [self.lemmatizer.lemmatize(t) for t in toks]
                if isinstance(toks, list) else []
            )

            return df

        except Exception as e:
            st.error(f"토큰화 오류: {str(e)}")
            return df

    def simple_sentiment_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """간단한 대체 감성 분석"""
        df_result = df.copy()

        positive_words = ['good', 'great', 'excellent', 'love', 'best', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor']

        def simple_score(text):
            if pd.isna(text):
                return 0.0
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            return (pos_count - neg_count) * 0.2

        df_result['sentiment_score'] = df_result['text_clean'].apply(simple_score)
        df_result['sentiment'] = df_result['sentiment_score'].apply(
            lambda s: 'Positive' if s > 0.1 else ('Negative' if s < -0.1 else 'Neutral')
        )

        return df_result

    # EVTireInsightAnalytics 클래스
    def reset_state(self):
        """상태 초기화"""
        self.use_default_data = False
        if 'reddit_df' in st.session_state:
            del st.session_state['reddit_df']

    def render(self):
        """Render EV Tire Insight Analytics UI"""
        st.header("🔋 EV Tire Insight Analytics")
        st.markdown("Reddit 데이터를 기반으로 EV 타이어 관련 인사이트를 분석합니다.")

        # Tab selection for data source
        data_tab1, data_tab2 = st.tabs(["📊 데이터 분석", "🔍 데이터 수집"])

        with data_tab2:
            # Reddit data collection interface
            collected_data = self.render_reddit_collector()

            if not collected_data.empty:
                st.markdown("---")
                st.subheader("🔄 수집된 데이터로 분석 진행")
                if st.button("수집된 데이터 분석하기"):
                    # Switch to analysis with collected data
                    df_processed = self.process_reddit_data(collected_data)
                    if not df_processed.empty:
                        st.success("✅ 수집된 데이터 분석 완료!")
                        # Continue with analysis...

        with data_tab1:
            # Original analysis interface
            # Check if default data file exists
            reddit_file_exists = check_file_exists(Config.EV_DATA_FILES["reddit"])

            # Data source selection
            col1, col2 = st.columns([3, 1])
            with col1:
                if reddit_file_exists:
                    st.success("✅ 기본 Reddit 데이터 파일이 준비되어 있습니다.")
                    self.use_default_data = True
                else:
                    st.info("📁 기본 Reddit 데이터 파일이 없습니다. 파일을 업로드하세요.")
                    self.use_default_data = False

            with col2:
                if st.button("🔄 초기화", key="ev_reset"):
                    self.use_default_data = False
                    st.rerun()

            # Load data
            df_processed = pd.DataFrame()

            if self.use_default_data and reddit_file_exists:
                # Load default data
                df_raw = DataProcessor.load_csv_safe(get_file_path(Config.EV_DATA_FILES["reddit"]))
                if not df_raw.empty:
                    df_processed = self.process_reddit_data(df_raw)
                    st.info(f"📊 기본 데이터 사용 중: {Config.EV_DATA_FILES['reddit']}")

            else:
                # File upload interface
                st.markdown("**Reddit 데이터 업로드**")
                st.caption("예: ev_tire_reddit_filtered.csv, reddit_comments.csv")
                uploaded_file = st.file_uploader("파일 선택", type=['csv'], key='reddit_file')

                if uploaded_file:
                    df_raw = DataProcessor.load_csv_from_upload(uploaded_file)
                    if not df_raw.empty:
                        df_processed = self.process_reddit_data(df_raw)

            # EV 분석 설정을 탭 내부에 배치
            if not df_processed.empty:
                # 1. 먼저 설정 부분
                with st.expander("🔧 EV 분석 설정", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        max_keywords = st.slider("키워드 분석 개수", 10, 50, 20)
                    with col2:
                        min_score = st.slider("최소 점수", 0, 100, 1)
                    with col3:
                        analysis_type = st.multiselect(
                            "분석 유형 선택",
                            ["키워드 분석", "감성 분석", "시간별 트렌드", "키워드 히트맵"],
                            default=["키워드 분석", "감성 분석"]
                        )

                # 2. 그 다음 데이터 처리 및 메트릭 표시
                try:
                    # Filter by score
                    df_filtered = df_processed[df_processed.get('score', 0) >= min_score]

                    # Display data info
                    st.success(f"✅ 데이터 처리 완료: {len(df_filtered):,}건")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 게시물", len(df_filtered))
                    with col2:
                        st.metric("서브레딧", df_filtered['subreddit'].nunique() if 'subreddit' in df_filtered.columns else 0)
                    with col3:
                        st.metric("기간", f"{df_filtered['created_utc'].dt.date.min()} ~ {df_filtered['created_utc'].dt.date.max()}")
                    with col4:
                        st.metric("평균 점수", f"{df_filtered.get('score', pd.Series([0])).mean():.1f}")

                    # 3. 분석 결과들
                    # Perform selected analyses
                    if "키워드 분석" in analysis_type:
                        st.subheader("🔍 키워드 분석 (TF-IDF)")

                        tfidf_results = self.perform_tfidf_analysis(df_filtered, max_keywords)

                        if not tfidf_results.empty:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                # Bar chart
                                fig = px.bar(
                                    tfidf_results.head(15),
                                    x='TF-IDF_Score' if 'TF-IDF_Score' in tfidf_results.columns else 'Frequency',
                                    y='Keyword',
                                    orientation='h',
                                    title="상위 키워드"
                                )
                                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.dataframe(tfidf_results.head(15), use_container_width=True)

                    if "감성 분석" in analysis_type:
                        st.subheader("😊 감성 분석")

                        df_sentiment = self.perform_sentiment_analysis(df_filtered)

                        if 'sentiment' in df_sentiment.columns:
                            # Sentiment distribution
                            sentiment_counts = df_sentiment['sentiment'].value_counts()

                            col1, col2 = st.columns(2)

                            with col1:
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="감성 분포"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.dataframe(sentiment_counts.to_frame('Count'), use_container_width=True)

                            # Sentiment over time
                            if "시간별 트렌드" in analysis_type:
                                st.subheader("📈 시간별 감성 트렌드")

                                df_sentiment['year_month'] = df_sentiment['created_utc'].dt.to_period('M')
                                monthly_sentiment = df_sentiment.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
                                monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0)

                                fig = px.line(
                                    monthly_sentiment_pct.reset_index(),
                                    x='year_month',
                                    y=monthly_sentiment_pct.columns,
                                    title="월별 감성 비율 변화"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    if "키워드 히트맵" in analysis_type and NLTK_AVAILABLE:
                        st.subheader("🔥 키워드 히트맵")

                        df_tokens = self.tokenize_and_lemmatize(df_filtered)

                        if 'tokens_lemmatized' in df_tokens.columns:
                            # Get top keywords
                            all_tokens = df_tokens['tokens_lemmatized'].explode().dropna()
                            top_keywords = all_tokens.value_counts().head(10).index.tolist()

                            if top_keywords:
                                # Create monthly heatmap data
                                df_expanded = df_tokens.explode('tokens_lemmatized')
                                df_top = df_expanded[df_expanded['tokens_lemmatized'].isin(top_keywords)].copy()
                                df_top['year_month'] = df_top['created_utc'].dt.to_period('M')

                                heatmap_data = (
                                    df_top
                                    .groupby(['year_month', 'tokens_lemmatized'])
                                    .size()
                                    .unstack(fill_value=0)
                                )

                                if not heatmap_data.empty:
                                    fig = px.imshow(
                                        heatmap_data.T,
                                        title="월별 키워드 빈도 히트맵",
                                        aspect="auto"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                    # Sample data display
                    with st.expander("📋 샘플 데이터"):
                        display_cols = ['created_utc', 'subreddit', 'score', 'text_clean']
                        available_cols = [col for col in display_cols if col in df_filtered.columns]
                        st.dataframe(df_filtered[available_cols].head(20))

                    # Download processed data
                    if st.button("📥 처리된 데이터 다운로드"):
                        csv = df_filtered.to_csv(index=False)
                        st.download_button(
                            label="CSV 다운로드",
                            data=csv,
                            file_name=f"processed_reddit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                    st.text(traceback.format_exc())

            else:
                if not self.use_default_data:
                    st.info("📁 Reddit 데이터 CSV 파일을 업로드하거나 '데이터 수집' 탭에서 새로 수집해주세요.")

                # Show expected data format
                with st.expander("📋 예상 데이터 형식"):
                    sample_data = pd.DataFrame({
                        'id': ['post1', 'post2'],
                        'created_utc': ['2024-01-01 12:00:00', '2024-01-02 13:00:00'],
                        'subreddit': ['electricvehicles', 'tires'],
                        'title': ['EV tire noise issues', 'Best tires for EVs'],
                        'body': ['Content about tire noise...', 'Recommendations for EV tires...'],
                        'score': [15, 23],
                        'num_comments': [5, 8]
                    })
                    st.dataframe(sample_data)

# ========================================
# APPLICATION ENTRY POINT
# ========================================
def main():
    """Main application entry point"""
    # Configure Streamlit
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Ensure data folder exists
    ensure_data_folder()

    # Main title and description
    st.title(Config.APP_TITLE)
    st.markdown(f"**Version {Config.VERSION}** | 통합 분석 대시보드")

    # Show data folder status
    with st.expander("📁 데이터 폴더 상태"):
        st.markdown(f"**데이터 폴더 경로:** `{Config.DATA_FOLDER}`")

        # Fleet TCO files
        st.markdown("**Fleet TCO 파일:**")
        for key, filename in Config.FLEET_DATA_FILES.items():
            exists = check_file_exists(filename)
            status = "✅" if exists else "❌"
            st.markdown(f"- {status} {filename}")

        # TBR files
        st.markdown("**TBR Market 파일:**")
        for key, filename in Config.TBR_DATA_FILES.items():
            exists = check_file_exists(filename)
            status = "✅" if exists else "❌"
            st.markdown(f"- {status} {filename}")

        # EV files
        st.markdown("**EV Analytics 파일:**")
        for key, filename in Config.EV_DATA_FILES.items():
            exists = check_file_exists(filename)
            status = "✅" if exists else "❌"
            st.markdown(f"- {status} {filename}")

    # Simplified sidebar - 공통 정보만
    with st.sidebar:
        st.header("📊 대시보드 정보")
        st.markdown("""
        ### 포함된 분석 도구:
        - **Fleet TCO Calculator**: 차량 운영 총비용 계산
        - **TBR Market Dashboard**: 글로벌 TBR 시장 분석
        - **EV Tire Analytics**: EV 타이어 인사이트 분석
        
        ### 지원 파일 형식:
        - CSV (UTF-8, UTF-8-BOM, CP949, EUC-KR)
        - Excel (XLSX, XLS)
        - SQLite (TBR 분석용)
        
        ### 기본 데이터 파일:
        데이터 폴더에 해당 이름의 파일을 저장하면 자동으로 로드됩니다.
        """)

        # Library status
        st.markdown("### 📦 라이브러리 상태:")
        if NLTK_AVAILABLE:
            st.success("✅ NLTK 설치됨")
        else:
            st.error("❌ NLTK 미설치")
            st.code("pip install nltk scikit-learn")

        if REDDIT_AVAILABLE:
            st.success("✅ PRAW 설치됨")
        else:
            st.error("❌ PRAW 미설치")
            st.code("pip install praw tqdm")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs([
        "🚛 Fleet TCO Calculator",
        "🌐 TBR Market Dashboard",
        "🔋 EV Tire Analytics"
    ])

    with tab1:
        try:
            calculator = FleetTCOCalculator()
            calculator.render()
        except Exception as e:
            st.error(f"Fleet TCO Calculator 오류: {str(e)}")

    with tab2:
        try:
            dashboard = TBRMarketDashboard()
            dashboard.render()
        except Exception as e:
            st.error(f"TBR Market Dashboard 오류: {str(e)}")

    with tab3:
        try:
            analytics = EVTireInsightAnalytics()
            analytics.render()
        except Exception as e:
            st.error(f"EV Tire Analytics 오류: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🚀 Integrated Analytics Dashboard | 
        Built with Streamlit | 
        Data-driven insights for transportation industry
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()  # 이 부분이 있는지 확인!
