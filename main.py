import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
import os 
import joblib
from datetime import datetime 



# --- Set up Streamlit page ---
st.set_page_config(layout="wide")
st.markdown("""
    <div style='font-size: 20px; font-weight: bold; color: #333333; margin-bottom: -8rem;'>
        AI SOLUTIONS DASHBOARD
    </div>
""", unsafe_allow_html=True)



# --- Increase file upload size limit ---
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "300"

report_df = pd.read_csv("classification_report.csv", index_col=0)




# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "view" not in st.session_state:
    st.session_state["view"] = "MANAGERIAL VIEW"

# --- Users and hashed passwords ---
users = {
    "administrator": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()),
    "AmantleG": bcrypt.hashpw("AmantleG123".encode(), bcrypt.gensalt()),
    "LoratoM": bcrypt.hashpw("LoratoM123".encode(), bcrypt.gensalt()),
    "MichelleT": bcrypt.hashpw("MichelleT".encode(), bcrypt.gensalt()),
    "KaoneW": bcrypt.hashpw("KaoneW123".encode(), bcrypt.gensalt()),
    "BoemoM": bcrypt.hashpw("BoemoM123".encode(), bcrypt.gensalt()),
    "TshepangC": bcrypt.hashpw("TshepangC123".encode(), bcrypt.gensalt()),
    "LetsoB": bcrypt.hashpw("LestoB123".encode(), bcrypt.gensalt()),
    "BotsileM": bcrypt.hashpw("BotsileM123".encode(), bcrypt.gensalt())
}

# --- Login Logic ---
def login():
    if not st.session_state["logged_in"]:
        login_form()
        st.stop()

def logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.rerun()

def login_form():
    st.title("AI SOLUTIONS DASHBOARD LOGIN")
    st.write("Please log in to continue.")

    email = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if email in users and bcrypt.checkpw(password.encode(), users[email]):
            st.session_state["logged_in"] = True
            st.session_state["user"] = email
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid email or password.")
    else:
        st.stop()

# --- Load and clean data ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, compression='gzip')
    df['User Agent'].fillna("Unknown", inplace=True)
    df['User ID'].fillna("Anonymous", inplace=True)
    df['Error Message'].fillna("None", inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour of Day'] = df['Timestamp'].dt.hour
    df['Day of Week'] = df['Timestamp'].dt.day_name()
    df['Weekend'] = df['Day of Week'].isin(['Saturday', 'Sunday'])
    df['Time Spent (minutes)'] = df['Session Duration (s)'] / 60
    df['Time Spent Category'] = pd.cut(df['Time Spent (minutes)'], bins=[0, 1, 5, 10, 30], labels=['<1 min', '1-5 min', '5-10 min', '10+ min'])
    return df


# Load preprocessing pipeline and model
@st.cache_resource
def load_model_and_preprocessor():
    preprocessor = joblib.load("preprocessing_pipeline.pkl")
    model = joblib.load("random_forest_model.pkl")
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()


model_features = [
    'Session Duration (s)',
    'Time on Product Page',
    'Total Visits',
    'Page Type',
    'Conversion Funnel Stage',
    'Event Type',
    'Device Type',
    'Referrer Type',
    'Requested Product',
    'Country',
     'Day of Week','Hour of Day', 'Weekend',
     'Demo Product',
    'Product Price', 'Product Discount', 'Transaction Amount',
    'First Visit'
]


def features(input_df):
    return input_df[model_features]



def main():
    login()
    st.markdown(
        """
        <style>
    
            svg {
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 2px 2px 5px rgba(1, 0, 0, 0.3);
                border: 3px solid #ADD8E6;
            }

            svg text {
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }

            /* Table Container Styling */
            .stDataFrame, .stTable {
                border-radius: 10px;
                box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2);
                border: 2px solid #ADD8E6;
                overflow: hidden;
                padding: 8px;
                background-color: #f9f9f9;
            }

            /* Optional: Add font and spacing for better table appearance */
            .stDataFrame table, .stTable table {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
            .kpi-box {
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }

            .kpi-title {
                font-size: 18px;
                color: white;
                margin-bottom: 0.3rem;
            }

            .kpi-value {
                font-size: 36px;
                font-weight: bold;
                color: white;
            }

            .kpi-subtext {
                font-size: 14px;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            /* Reduce top and bottom padding of the main container */
            .block-container {
                padding-top: 0.3rem;
                padding-bottom: 0.3rem;
            }
    
            /* Reduce vertical spacing between elements */
            .stMetric, .stPlotlyChart, .stSubheader, .stMarkdown {
                margin-bottom: 0.3rem !important;
            }
    
            /* Remove extra spacing between rows */
            .element-container {
                padding-bottom: 0.2rem !important;
            }
    
            /* Optional: tighten up column spacing */
            .stColumns {
                gap: 0.5rem !important;
            }
        </style>
            """, unsafe_allow_html=True)


    st.markdown("""
        <style>
            .block-container {
                margin-top: -2rem;
                margin-bottom: -6rem;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
                
            h4 {
                margin-bottom: 0.3rem;  /* Reduce space below title */
                margin-top: 0.8rem;     /* Optional: reduce space above title */
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar: Welcome box ---
    st.sidebar.markdown(
        f"""
        <div style="
            background-color: #2c3e50;
            padding: 16px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        ">
            <div style="font-size: 18px; font-weight: bold;">👋 Welcome!</div>
            <div style="margin-top: 8px;">Logged in as:<br><b>{st.session_state['user']}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Upload Data ---
    st.sidebar.markdown("### 📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state["data"] = load_data(uploaded_file)

    # --- Sidebar View Buttons ---
    st.sidebar.markdown("Navigation", unsafe_allow_html=True)
    view_options = ["MANAGERIAL VIEW", "SALES TEAM VIEW", "SALES AND TRAFFIC DATA", "ADVERTISEMENT TEAM VIEW","SESSION OUTCOME PREDICTOR"]
    for option in view_options:
        button_class = "sidebar-button sidebar-selected" if st.session_state.view == option else "sidebar-button"
        if st.sidebar.button(option, key=option, use_container_width=True):
            st.session_state.view = option

    # --- Proceed only if data is loaded ---
    if "data" in st.session_state and st.session_state["data"] is not None:
        df = st.session_state["data"]

        with st.sidebar.expander("🔍 Filters", expanded=False):
            start_date = st.date_input("Start Date", df['Timestamp'].min().date())
            end_date = st.date_input("End Date", df['Timestamp'].max().date())

            # Country filter
            country_options = ["All"] + sorted(df['Country'].dropna().unique())
            selected_country = st.selectbox("Country", country_options, index=0)

            # Device filter
            device_options = ["All"] + sorted(df['Device Type'].dropna().unique())
            selected_device = st.selectbox("Device Type", device_options, index=0)

            # Campaign filter
            campaign_options = ["All"] + sorted(df['Campaign Type'].dropna().unique())
            selected_campaign = st.selectbox("Campaign Type", campaign_options, index=0)

            # Salesperson filter
            ##salesperson_options = ["All"] + sorted(df['Salesperson Name'].dropna().unique())
            ##selected_salesperson = st.selectbox("Salesperson Name", salesperson_options, index=0)

        df_filtered = df[
            (df['Timestamp'].dt.date >= start_date) &
            (df['Timestamp'].dt.date <= end_date)
        ]

        if selected_country != "All":
            df_filtered = df_filtered[df_filtered['Country'] == selected_country]

        if selected_device != "All":
            df_filtered = df_filtered[df_filtered['Device Type'] == selected_device]

        if selected_campaign != "All":
            df_filtered = df_filtered[df_filtered['Campaign Type'] == selected_campaign]

       
        
         # --- Filtered successful sales only ---
        sales_data = df_filtered[df_filtered["Sale Made"].str.lower() == "yes"]

        # --- Dynamic year detection from filtered data ---
        years_in_filtered = sales_data["Timestamp"].dt.year.unique()
        years_in_filtered.sort()

        if len(years_in_filtered) >= 2:
            this_year1 = years_in_filtered[-1]
            last_year1 = years_in_filtered[-2]
        elif len(years_in_filtered) == 1:
            this_year1 = years_in_filtered[0]
            last_year1 = this_year1 - 1
        else:
            # Fallback if no data
            this_year1 = pd.Timestamp.now().year
            last_year1 = this_year1 - 1
        



        # --- MANAGERIAL VIEW ---
        if st.session_state.view == "MANAGERIAL VIEW":
            st.markdown("<h4> MANAGERIAL VIEW </h4>", unsafe_allow_html=True)

            st.markdown("""
                <style>
                    h6 {
                        margin-bottom: 0.2rem;
                    }
                    .element-container:has(h6) + div {
                        margin-top: -10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            # --- Prepare date/time related data ---
            today = pd.Timestamp.now()
            current_year = this_year1
            current_month = today.month
            
            # For your example, let's explicitly use March and Feb of current year
            target_month = current_month  # March
            prev_month = target_month - 1 if target_month > 1 else 12
            prev_month_year = current_year if target_month > 1 else current_year - 1

            # Monthly revenue for March (current year)
            revenue_march = df_filtered[
                (df_filtered['Timestamp'].dt.year == current_year) & 
                (df_filtered['Timestamp'].dt.month == target_month) &
                (df_filtered['Sale Made'].str.lower() == 'yes')
            ]['Transaction Amount'].sum()

            # Monthly revenue for previous month
            revenue_prev_month = df_filtered[
                (df_filtered['Timestamp'].dt.year == prev_month_year) & 
                (df_filtered['Timestamp'].dt.month == prev_month) &
                (df_filtered['Sale Made'].str.lower() == 'yes')
            ]['Transaction Amount'].sum()

            # Calculate delta and arrow color
            delta = revenue_march - revenue_prev_month
            delta_percent = (delta / revenue_prev_month * 100) if revenue_prev_month > 0 else 0
            arrow = "🔺" if delta >= 0 else "🔻"
            arrow_color = "green" if delta >= 0 else "red"
            tile_color = "green" if delta >= 0 else "red"

            # --- Define revenue targets ---
            ytd_target = (14300000)
            qtd_target = (3000000)

            # --- YTD revenue (Jan 1 - today) ---
            ytd_revenue = df_filtered[
                (df_filtered['Timestamp'].dt.year == current_year) &
                (df_filtered['Sale Made'].str.lower() == 'yes')
            ]['Transaction Amount'].sum()

            # --- QTD revenue ---
            qtd_revenue = df_filtered[
                (df_filtered['Timestamp'].dt.quarter == pd.Timestamp.now().quarter) &
                (df_filtered['Timestamp'].dt.year == current_year) &
                (df_filtered['Sale Made'].str.lower() == 'yes')
            ]['Transaction Amount'].sum()

            # --- Function: gauge color by thresholds ---
            def ytd_colorfun(value, ytd_target, tolerance=0.65):
                if value >= ytd_target:
                    return "green"
                elif value >= ytd_target * tolerance:
                    return "orange"
                else:
                    return "red"
                
            def qtd_colorfun(value, qtd_target, tolerance=0.65):
                if value >= qtd_target:
                    return "green"
                elif value >= qtd_target * tolerance:
                    return "orange"
                else:
                    return "red"
            
            
            # --- Color coding based on thresholds ---
            #ytd_thresholds = (50000, 150000)
            #qtd_thresholds = (15000, 50000)

            ytd_color = ytd_colorfun(ytd_revenue, ytd_target)
            qtd_color = qtd_colorfun(qtd_revenue, qtd_target)
            #month_color = month_colorfun(revenue_march, revenue_march)
            # --- Arrow logic for YTD ---
            ytd_arrow = "🔺" if ytd_revenue >= ytd_target else "🔻"
            ytd_arrow_color = "green" if ytd_revenue >= ytd_target else "red"

            # --- Arrow logic for QTD ---
            qtd_arrow = "🔺" if qtd_revenue >= qtd_target else "🔻"
            qtd_arrow_color = "green" if qtd_revenue >= qtd_target else "red"


            # --- Layout: Two Columns ---
            col1, col2 = st.columns([6,4])  

            # --- Column 1 Content ---
            with col1:

                st.markdown("<h6>📈 SALES PERFORMANCE KPI's</h6>", unsafe_allow_html=True)

                # --- Row 1: Three KPI Tiles ---
                kpi1_col, kpi2_col, kpi3_col = st.columns(3)

                with kpi1_col:
                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {tile_color}; height: 200px;">
                            <div class="kpi-title">MONTH REVENUE</div>
                            <div class="kpi-value">
                                ${revenue_march:,.0f}
                                <span style='font-size: 24px; color: {arrow_color}; position: relative; top: -2px; margin-left: 2px;'> {arrow}</span>
                            </div>
                            <div class="kpi-subtext">Previous month: ${revenue_prev_month:,.0f}</div>
                        </div>
                    """, unsafe_allow_html=True)


                with kpi2_col:
                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {ytd_color}; height: 200px;">
                            <div class="kpi-title">2025 REVENUE</div>
                            <div class="kpi-value">
                                ${ytd_revenue:,.0f} 
                                <span style= 'color': {ytd_arrow_color}; font-size: 24px;'>{ytd_arrow}</span>
                                <div class="kpi-subtext">Target: ${ytd_target:,.0f}</div>
                        </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with kpi3_col:
                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {qtd_color}; height: 200px;">
                            <div class="kpi-title">2025 Q2 REVENUE</div>
                            <div class="kpi-value">
                                ${qtd_revenue:,.0f}
                                <span style=' color: {qtd_arrow_color}; font-size: 24px;'>{qtd_arrow}</span>
                                <div class="kpi-subtext">Target: ${qtd_target:,.0f}</div>
                        </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)


                # --- Row 2: Line Chart ---
                
                 
                st.markdown("<h6>2024 VS 2025 REVENUE COMPARISON</h6>", unsafe_allow_html=True)


                # Prepare data
                df_sales = df_filtered[df_filtered['Sale Made'].str.lower() == 'yes'].copy()
                df_sales['Year'] = df_sales['Timestamp'].dt.year
                df_sales['Month'] = df_sales['Timestamp'].dt.month
                revenue_by_month = df_sales.groupby(['Year', 'Month'])['Transaction Amount'].sum().reset_index()
                pivot_revenue = revenue_by_month.pivot(index='Month', columns='Year', values='Transaction Amount').fillna(0)

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=pivot_revenue.index, y=pivot_revenue.get(current_year, [0]*12), mode='lines+markers', name=str(current_year)))
                fig_line.add_trace(go.Scatter(x=pivot_revenue.index, y=pivot_revenue.get(current_year-1, [0]*12), mode='lines+markers', name=str(current_year-1)))

                fig_line.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Revenue ($)",
                    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
                    height=150,
                    margin=dict(t=20, b=40)
                )

                st.plotly_chart(fig_line, use_container_width=True)


            # --- Column 2 Content: Leaderboard ---
            with col2:
                
                st.markdown("<h6>SALES TEAM LEADERBOARD</h6>", unsafe_allow_html=True)

                # Filter only sales from the current year
                df_current_year_sales = df_filtered[
                    (df_filtered['Sale Made'].str.lower() == 'yes') &
                    (df_filtered['Timestamp'].dt.year == current_year)
                ]
                
                # Group by Salesperson and aggregate total revenue and number of sales
                leadership_table = df_current_year_sales.groupby('Salesperson Name').agg({
                    'Transaction Amount': 'sum',
                    'Sale Made': 'count'
                }).reset_index()


                # Rename columns
                leadership_table.columns = ['Name', 'Total Revenue', 'Number of Sales Made']

                # Sort by Total Revenue descending
                leadership_table = leadership_table.sort_values(by='Total Revenue', ascending=False)

                # Format currency and set display height
                st.dataframe(
                    leadership_table.style.format({
                        'Total Revenue': '£{:.2f}'
                    }),
                    use_container_width=True,
                    height=350,
                    hide_index=True
                )
                    

            # --- SALES TEAM VIEW ---
        elif st.session_state.view == "SALES TEAM VIEW":

            
            st.markdown("""
                <style>
                    h6 {
                        margin-bottom: 0.2rem;
                    }
                    .element-container:has(h6) + div {
                        margin-top: -10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)
          
            st.markdown("<h4> SALES TEAM MEMBERS PERFORMANCE </h4>", unsafe_allow_html=True)

            salesperson_options = ["All"] + sorted(df_filtered['Salesperson Name'].dropna().unique())
            selected_salesperson = st.selectbox("FILTER BY SALESPERSON", salesperson_options, index=0, key="salesperson_filter")

            df_sales_team = df_filtered.copy()

            # Filter only successful sales
            sales_data = df_filtered[df_filtered["Sale Made"].str.lower() == "yes"]

            # Further filter by selected salesperson if applicable
            if selected_salesperson != "All":
                sales_data = sales_data[sales_data["Salesperson Name"] == selected_salesperson]

            # Ensure Year and Month columns exist on filtered data
            sales_data["Year"] = sales_data["Timestamp"].dt.year
            sales_data["Month"] = sales_data["Timestamp"].dt.month


#----------METRICS----------
            today = pd.Timestamp.today()
            this_year = this_year1
            last_year = last_year1
            this_month = today.month

            # Ensure Timestamp is datetime
            df_filtered["Timestamp"] = pd.to_datetime(df_filtered["Timestamp"])

            # Extract year and month
            df_filtered["Year"] = df_filtered["Timestamp"].dt.year
            df_filtered["Month"] = df_filtered["Timestamp"].dt.month

            # Filter only successful sales
            # Further filter by selected salesperson if applicable
            

            # Group by year/month and sum transaction amounts
            monthly_data = sales_data.groupby(["Year", "Month"]).agg({"Transaction Amount": "sum"}).reset_index()
            monthly_data = monthly_data.rename(columns={"Transaction Amount": "Revenue"})

            # Convert numeric month to month name for display
            monthly_data["Month"] = monthly_data["Month"].apply(lambda x: datetime.strptime(str(x), "%m").strftime("%b"))

            # Order month names properly
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly_data["Month"] = pd.Categorical(monthly_data["Month"], categories=month_order, ordered=True)

            # Split data by year
            monthly_data_this_year = monthly_data[monthly_data["Year"] == this_year].sort_values("Month")
            monthly_data_last_year = monthly_data[monthly_data["Year"] == last_year].sort_values("Month")

            # Team average (3-month rolling)
            team_average = monthly_data_this_year["Revenue"].rolling(3, min_periods=1).mean()

            # Revenue This year vs Last year
            revenue_this_year = sales_data[sales_data["Timestamp"].dt.year == this_year] ["Transaction Amount"].sum()
                                           
            revenue_last_year = sales_data[sales_data["Timestamp"].dt.year == last_year] ["Transaction Amount"].sum()


            # Revenue this month
            revenue_this_month = sales_data[
                (sales_data["Timestamp"].dt.year == this_year) &
                (sales_data["Timestamp"].dt.month == this_month)
            ]["Transaction Amount"].sum()


            # Revenue last month
            revenue_last_month = sales_data[
                (sales_data["Timestamp"].dt.year == this_year) &
                (sales_data["Timestamp"].dt.month == this_month -1)
            ]["Transaction Amount"].sum()


            # YTD and QTD
            revenue_ytd = sales_data[sales_data["Timestamp"].dt.year == this_year]["Transaction Amount"].sum()
            ##sales_ytd = df_filtered[df_filtered["Timestamp"].dt.year == this_year]["Requested Product"].count()

            


            sales_ytd= sales_data[(sales_data['Sale Made'].str.lower() == 'yes') & (df_filtered['Timestamp'].dt.year == this_year)].shape[0]

            revenue_qtd = sales_data[(sales_data["Timestamp"].dt.quarter == pd.Timestamp.now().quarter) & 
                                    (sales_data["Timestamp"].dt.year == this_year)]["Transaction Amount"].sum()

            # Row 1
            col1, col2 = st.columns([1, 3])

            # --- Column 1: KPI Tile ---
            with col1:
                delta = revenue_this_month - revenue_last_month
                arrow = "🔺" if delta > 0 else "🔻"
                arrow_color = "green" if delta > 0 else "red"

                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {arrow_color};">
                        <div class="kpi-title"> Revenue This Month</div>
                        <div class="kpi-value">
                            ${revenue_this_month:,.0f} <span style="font-size: 28px;">{arrow}</span>
                        </div>
                        <div class="kpi-subtext">Last month: ${revenue_last_month:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)


            # --- Column 2: Triple Gauge ---
            with col2:
                kpi1, kpi2, kpi3 = st.columns(3)


                # --- Sales YTD
                # --- Sales YTD Thresholds by Salesperson ---
                sales_targets = {
                    "Amantle": (4000, 7000),
                    "Lorato": (4000, 7500),
                    "Kaone": (4000, 8000),
                    "Letso": (5000, 8200),
                    "Bostile": (4000, 8500),
                    "Boemo": (8300, 8700),
                    "Michelle": (4000, 8500),
                    "Tshepang": (5000, 8000)
                }

                # Get thresholds for the selected salesperson
                lower_threshold, upper_threshold = sales_targets.get(selected_salesperson, (30000, 70000 ))  # fallback default
                sales_goal = upper_threshold

                # --- Determine Background Color ---
                if sales_ytd < lower_threshold:
                    bg_sales = "red"
                elif sales_ytd < upper_threshold:
                    bg_sales = "orange"
                else:
                    bg_sales = "green"


                with kpi1:
                    arrow = "🔺" if sales_ytd >= sales_goal else "🔻"
                    arrow_color = "green" if sales_ytd >= sales_goal else "red"

                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {bg_sales};">
                            <div class="kpi-title"> PRODUCT SALES</div>
                            <div class="kpi-value">{sales_ytd:,.0f} <span style="color: {arrow_color};">{arrow}</span>
                            </div>
                            <div class="kpi-subtext">Goal: {sales_goal:,.0f}</div>
                        </div>
                    """, unsafe_allow_html=True)




                # --- Revenue YTD
                yearlyR_targets = {
                    "Amantle": (900000, 1000000),
                    "Lorato": (700000, 1000000),
                    "Kaone": (600000, 1200000),
                    "Letso": (500000, 1100000),
                    "Bostile": (600000, 1200000),
                    "Boemo": (700000, 140000),
                    "Michelle": (700000, 1300000),
                    "Tshepang": (500000, 1500000)
                }

                # Get thresholds for the selected salesperson
                lower_threshold, upper_threshold = yearlyR_targets.get(selected_salesperson, (7000000,14300000 ))  # fallback default
                annual_targets= upper_threshold

                    # --- Determine Background Color ---
                if revenue_ytd < lower_threshold:
                    bg_revenue_ytd = "red"
                elif revenue_ytd < upper_threshold:
                    bg_revenue_ytd = "orange"
                else:
                    bg_revenue_ytd = "green"

            
                with kpi2:
                    
                    arrow = "🔺" if revenue_ytd >= annual_targets else "🔻"
                    arrow_color = "green" if revenue_ytd >= annual_targets else "red"

                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {bg_revenue_ytd};">
                            <div class="kpi-title">💵 ANNUAL REVENUE</div>
                            <div class="kpi-value">${revenue_ytd:,.0f} <span style="color: {arrow_color};">{arrow}</span></div>
                            <div class="kpi-subtext">Target: ${annual_targets:,.0f} 
                            </div>
                        </div>
                    """, unsafe_allow_html=True)




                    # --- Revenue QTD
                quarterlY_targets = {
                    "Amantle": (270800, 390000),
                    "Lorato": (148200, 350000),
                    "Kaone": (233333, 400000),
                    "Letso": (300000, 410000),
                    "Bostile": (200000, 420000),
                    "Boemo": (125000, 400000),
                    "Michelle": (233330, 430000),
                    "Tshepang": (260000, 400000)

                    }
                
                lower_threshold, upper_threshold = quarterlY_targets.get(selected_salesperson, ( 2500000,3000000))  # fallback default
                quarterly_target = upper_threshold


                    # --- Determine Background Color ---
                if revenue_qtd < lower_threshold:
                    bg_revenue_qtd = "red"
                elif revenue_qtd < upper_threshold:
                    bg_revenue_qtd = "orange"
                else:
                    bg_revenue_qtd = "green"

                # Get thresholds for the selected salesperson



                with kpi3:
                    
                    arrow = "🔺" if revenue_qtd >= quarterly_target else "🔻"
                    arrow_color = "green" if revenue_qtd >= quarterly_target else "red"

                    st.markdown(f"""
                        <div class="kpi-box" style="background-color: {bg_revenue_qtd};">
                            <div class="kpi-title">📆 QUARTERLY REVENUE</div>
                            <div class="kpi-value">${revenue_qtd:,.0f}<span style="color: {arrow_color};">{arrow}</span></div>
                            <div class="kpi-subtext">Target: ${quarterly_target:,.0f} 
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        
        
        #------------ Row 2
            col3, col4 = st.columns([3, 1])

            # --- Column 1: Line Chart ---
            with col3:
                
                st.markdown("<h6>📈 Product Sales: 2025 vs 2024</h6>", unsafe_allow_html=True)

                team_avg_aligned = monthly_data_this_year.set_index("Month").reindex(month_order)["Revenue"].rolling(3, min_periods=1).mean().fillna(0)


    
                df_chart = pd.DataFrame({
                    "Month": month_order,
                    "This Year": monthly_data_this_year.set_index("Month").reindex(month_order)["Revenue"].fillna(0).values,
                    "Last Year": monthly_data_last_year.set_index("Month").reindex(month_order)["Revenue"].fillna(0).values,
                    "Team Avg": team_avg_aligned.values
                })

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["This Year"], name="This Year", mode="lines+markers"))
                fig_line.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["Last Year"], name="Last Year", mode="lines+markers"))
                fig_line.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["Team Avg"], name="Team Avg", mode="lines+markers", line=dict(dash="dot", color="gray")))
                fig_line.update_layout(
                                    xaxis_title="Month", yaxis_title="Revenue",
                                    legend_title="Year")
                fig_line.update_layout(
                    
                    height=200,
                    margin=dict(t=20, b=40)
                )
                st.plotly_chart(fig_line, use_container_width=True)

            # --- Column 2: Personalized Engagement Tile ---
            # yearly revenue targets by salesperson
        
            year_targets = {
                "Amantle": (1000000),
                "Lorato": (1000000),
                "Kaone": (1200000),
                "Letso": (1100000),
                "Bostile": (1200000),
                "Boemo": (140000),
                "Michelle": (1300000),
                "Tshepang": (1500000)
            }

            # Get the target for the selected salesperson
            targets = year_targets.get(selected_salesperson, 14300000)  # fallback default if name not matched

            with col4:
                st.markdown("#### MOTIVATION")
                if revenue_this_year > targets:
                    st.markdown(f"""
                        <div style="background-color:#d4edda; color:#155724; padding:20px; border-radius:10px; font-size:20px; min-height:120px;">
                            🔥 <b>How exciting!! You're smashing your yearly target!!keep it up!</b>
                        </div>
                    """, unsafe_allow_html=True)

                elif revenue_this_year > revenue_last_year:
                    st.markdown(f"""
                        <div style="background-color:#d1ecf1; color:#0c5460; padding:20px; border-radius:10px; font-size:20px; min-height:120px;">
                            📈 <b>You're growing, keep up the great momentum!</b>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                        <div style="background-color:#fff3cd; color:#856404; padding:20px; border-radius:10px; font-size:20px; min-height:120px;">
                            📊 <b>Stay focused, we believe in you and you’re close to turning it around!</b>
                        </div>
                    """, unsafe_allow_html=True)
        
        elif st.session_state.view == "SALES AND TRAFFIC DATA":
            st.markdown("""
                <style>
                    h6 {
                        margin-bottom: 0.2rem;
                    }
                    .element-container:has(h6) + div {
                        margin-top: -10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)


            st.markdown("<h4>SALES AND TRAFFIC DATA </h4>", unsafe_allow_html=True)

            # Ensure Timestamp is datetime
            df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'], errors='coerce')

            # Normalize Page Type values
            df_filtered['Page Type'] = df_filtered['Page Type'].str.lower().str.strip()

            # --- Total Visits ---
            total_visits_this_year = df_filtered[df_filtered['Timestamp'].dt.year == this_year1].shape[0]
            total_visits_last_year = df_filtered[df_filtered['Timestamp'].dt.year == last_year1].shape[0]

            # --- Product Page Visits ---
            product_visits_this_year = df_filtered[
                (df_filtered['Timestamp'].dt.year == this_year1) & 
                (df_filtered['Page Type'] == 'product page')
            ].shape[0]

            product_visits_last_year = df_filtered[
                (df_filtered['Timestamp'].dt.year == last_year1) & 
                (df_filtered['Page Type'] == 'product page')
            ].shape[0]

            region_col = 'Region' if 'Region' in df_filtered.columns else 'Country'
            sales_by_region = df_filtered[df_filtered['Sale Made'].str.lower() == 'yes'][region_col].value_counts().reset_index()
            sales_by_region.columns = ['Country', 'Sales']


            # --- Total Sales This Year and Last Year ---
            sales_this_year = df_filtered[
                (df_filtered['Timestamp'].dt.year == this_year1) &
                (df_filtered['Sale Made'].str.lower() == 'yes')
            ].shape[0]

            # --- Targets ---
            visit_target = 200000
            product_visit_target = 150000
            sales_target = 60000

            def get_performance_color(value, target, threshold=0.6):
                if value >= target:
                    return "green"
                elif value >= target * (1 - threshold):
                    return "orange"
                else:
                    return "red"
                
            visit_color = get_performance_color(total_visits_this_year, visit_target)
            product_color = get_performance_color(product_visits_this_year, product_visit_target)
            sales_color = get_performance_color(sales_this_year, sales_target)

            visit_arrow = "🔺" if total_visits_this_year >= visit_target else "🔻"
            product_arrow = "🔺" if product_visits_this_year >= product_visit_target else "🔻"
            sales_arrow = "🔺" if sales_this_year >= sales_target else "🔻"



            kpi_row= st.columns(3)
                            # --- KPI Boxes ---
            with kpi_row[0]:
                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {visit_color}; padding: 15px; border-radius: 10px; color: white;">
                        <div class="kpi-title"> Total Website Visits</div>
                        <div class="kpi-value">{total_visits_this_year:,} <span style="font-size: 24px;">{visit_arrow}</span></div>
                        <div class="kpi-subtext">Target: {visit_target:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_row[1]:
                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {product_color}; padding: 15px; border-radius: 10px; color: white;">
                        <div class="kpi-title"> Product Page Visits</div>
                        <div class="kpi-value">{product_visits_this_year:,} <span style="font-size: 24px;">{product_arrow}</span></div>
                        <div class="kpi-subtext">Target: {product_visit_target:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_row[2]:
                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {sales_color}; padding: 15px; border-radius: 10px; color: white;">
                        <div class="kpi-title"> Sales Count</div>
                        <div class="kpi-value">{sales_this_year:,} <span style="font-size: 24px;">{sales_arrow}</span></div>
                        <div class="kpi-subtext">Target: {sales_target:,}</div>
                    </div>
                """, unsafe_allow_html=True)




                ## st.markdown("---")

            # Filter data for the current year
            df_current_year = df_filtered[df_filtered['Timestamp'].dt.year == this_year1]

            # --- Row 1: Choropleth + Sales by Day of Week ---
            row1_col1, row1_col2, row1_col3 = st.columns([3,4,3])

            with row1_col1:
                
                st.markdown("<h6> Sales by Country</h6>", unsafe_allow_html=True)
                fig_sales_by_region = px.choropleth(
                    sales_by_region,
                    locations='Country',
                    locationmode='country names',
                    color='Sales',
                    labels={'Sales': 'Sales'}
                )
                fig_sales_by_region.update_geos(scope='europe')
                fig_sales_by_region.update_layout(height=150, margin=dict(t=30, b=30))
                st.plotly_chart(fig_sales_by_region, use_container_width=True)


            with row1_col2:
                st.markdown("<h6> Sales by Day of Week</h6>", unsafe_allow_html=True)
                sales_by_day = df_current_year.groupby(df_current_year['Timestamp'].dt.day_name()).size().reset_index(name='Sales')
                sales_by_day.columns = ['Day of Week', 'Sales']
                fig_sales_by_day = px.bar(sales_by_day, x='Day of Week', y='Sales')
                fig_sales_by_day.update_layout(height=150, margin=dict(t=10, b=10))
                st.plotly_chart(fig_sales_by_day, use_container_width=True)

            # --- Row 2: Sales by Hour + Common Products ---
            #row2_col1, row2_col2 = st.columns(2)

            with row1_col3:
                
                st.markdown("<h6> Peak Traffic Hours</h6>", unsafe_allow_html=True)
                sales_by_hour = df_current_year.groupby(df_current_year['Timestamp'].dt.hour).size().reset_index(name='Sales')
                sales_by_hour.columns = ['Hour of Day', 'Sales']
                fig_sales_by_hour = px.line(sales_by_hour, x='Hour of Day', y='Sales')
                fig_sales_by_hour.update_layout(height=150, margin=dict(t=10, b=10))
                st.plotly_chart(fig_sales_by_hour, use_container_width=True)
            # --- Row 2: Device, Traffic Source, Browser Distribution ---
            # --- Row 2: Device, Traffic Source, Browser Distribution ---

            df_sales_current_year = df_current_year[df_current_year['Sale Made'].str.lower() == 'yes']
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row2_col1:
                st.markdown("<h6>Device Distribution</h6>", unsafe_allow_html=True)
                device_counts = df_sales_current_year['Device Type'].value_counts().reset_index()
                device_counts.columns = ['Device Type', 'Count']
                fig_device = px.bar(
                    device_counts,
                    x='Device Type',
                    y='Count',
                    color='Device Type',
                    text='Count'
                )
                fig_device.update_layout(height=150, margin=dict(t=10, b=10), showlegend=False)
                fig_device.update_traces(textposition='outside')
                st.plotly_chart(fig_device, use_container_width=True)

            with row2_col2:
                st.markdown("<h6>Traffic Source Distribution</h6>", unsafe_allow_html=True)
                traffic_counts = df_sales_current_year['Referrer Type'].value_counts().reset_index()
                traffic_counts.columns = ['Referrer Type', 'Count']
                fig_traffic = px.pie(traffic_counts, names='Referrer Type', values='Count', hole=0.4)
                fig_traffic.update_layout(height=150, margin=dict(t=10, b=10))
                st.plotly_chart(fig_traffic, use_container_width=True)

            with row2_col3:
                st.markdown("<h6>Browser Distribution</h6>", unsafe_allow_html=True)
                browser_counts = df_sales_current_year['User Agent'].value_counts().reset_index()
                browser_counts.columns = ['User Agent', 'Count']
                fig_browser = px.pie(browser_counts, names='User Agent', values='Count', hole=0.4)
                fig_browser.update_layout(height=150, margin=dict(t=10, b=10))
                st.plotly_chart(fig_browser, use_container_width=True)



        elif st.session_state.view == "ADVERTISEMENT TEAM VIEW":

            st.markdown("<h4> ADVERTISEMENT TEAM VIEW </h4>", unsafe_allow_html=True)
            st.markdown("""
                <style>
                    h6 {
                        margin-bottom: 0.4rem;
                    }
                    .element-container:has(h6) + div {
                        margin-top: -10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            # --- Time Filtering ---
            df_filtered['Transaction Date'] = pd.to_datetime(df_filtered['Transaction Date'])
            this_year = pd.Timestamp.now().year
            last_year = this_year - 1
            df_filtered['Year'] = df_filtered['Transaction Date'].dt.year
            df_filtered['Month'] = df_filtered['Transaction Date'].dt.strftime('%b')

            df_this_year = df_filtered[df_filtered['Year'] == this_year]
            #df_last_year = df_filtered[df_filtered['Year'] == last_year]


            avg_time_target = 350.00  # in minutes
            demo_target = 30000       # target number of demo requests

            def get_performance_color(value, target, threshold=0.1):  # use 0.1 for 10% threshold
                if value >= target:
                    return "green"
                elif value >= target * (1 - threshold):
                    return "orange"
                else:
                    return "red"

            # --- KPIs ---
            avg_time_this = df_this_year['Time on Product Page'].mean()
            demo_this = df_this_year[df_this_year['Demo Requested'].str.lower() == 'yes'].shape[0]

            # --- Styling ---
            avg_time_color = get_performance_color(avg_time_this, avg_time_target)
            demo_requests_color = get_performance_color(demo_this, demo_target)

            avg_time_arrow = "🔺" if avg_time_this >= avg_time_target else "🔻"
            demo_requests_arrow = "🔺" if demo_this >= demo_target else "🔻"

            # --- Display Styled KPIs ---
            kpi1, kpi2 = st.columns(2)


            with kpi1:
                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {avg_time_color}; padding: 15px; border-radius: 10px; color: white;">
                        <div class="kpi-title" style="font-size: 16px; font-weight: bold;"> Average Time on Product Pages</div>
                        <div class="kpi-value" style="font-size: 28px; margin-top: 5px;">{avg_time_this:.2f}m <span style="font-size: 24px;">{avg_time_arrow}</span></div>
                        <div class="kpi-subtext" style="font-size: 14px; margin-top: 5px;">Target: {avg_time_target:.2f}m</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi2:
                st.markdown(f"""
                    <div class="kpi-box" style="background-color: {demo_requests_color}; padding: 15px; border-radius: 10px; color: white;">
                        <div class="kpi-title" style="font-size: 16px; font-weight: bold;"> Demo Requests</div>
                        <div class="kpi-value" style="font-size: 28px; margin-top: 5px;">{demo_this:,} <span style="font-size: 24px;">{demo_requests_arrow}</span></div>
                        <div class="kpi-subtext" style="font-size: 14px; margin-top: 5px;">Target: {demo_target:,}</div>
                    </div>
                """, unsafe_allow_html=True)




            # === ROW 2 ===
            row2_col1, row2_col2 = st.columns(2)

            # Demo Requests by Country and Product Type
            with row2_col1:
                
                st.markdown("<h6>Demo Requests by Country & Product Type</h6>", unsafe_allow_html=True)
                demo_df = df_filtered[df_filtered['Demo Requested'].str.lower() == 'yes']
                demo_group = demo_df.groupby(['Country', 'Demo Product']).size().reset_index(name='Count')
                fig1 = go.Figure()
                for product in demo_group['Demo Product'].unique():
                    subset = demo_group[demo_group['Demo Product'] == product]
                    fig1.add_trace(go.Bar(x=subset['Country'], y=subset['Count'], name=product))
                fig1.update_layout(
                    barmode='stack',
                    height=150,
                    margin=dict(t=30, b=40),
                    xaxis_title="Country", yaxis_title="Count"
                )
                st.plotly_chart(fig1, use_container_width=True)

            # Correlation between Demo Requests and Sales
            with row2_col2:
                st.markdown("<h6> Demo Requests vs Product Requests</h6>", unsafe_allow_html=True)

                # Use the already filtered df_this_year
                demo_counts = df_filtered[df_filtered['Demo Requested'].str.lower() == 'yes'].groupby("Demo Product").size()
                sales_counts = df_filtered[df_filtered['Sale Made'].str.lower() == 'yes'].groupby("Demo Product").size()

                # Combine and prepare dataframe
                corr_df = pd.concat([demo_counts, sales_counts], axis=1, keys=["Demo Requests", "Sales"]).fillna(0).reset_index()

                # Build bar chart
                fig2 = go.Figure(data=[
                    go.Bar(name='Demo Requests', x=corr_df['Demo Product'], y=corr_df['Demo Requests']),
                    go.Bar(name='Product Requests', x=corr_df['Demo Product'], y=corr_df['Sales'])
                ])

                fig2.update_layout(
                    barmode='group',
                    height=150,
                    margin=dict(t=30, b=40),
                    xaxis_title="Demo Product",
                    yaxis_title="Count"
                )

                st.plotly_chart(fig2, use_container_width=True)

            # === ROW 3 ===
            row3_col1, row3_col2 = st.columns(2)

            # Monthly Demo Requests Over Time
            with row3_col1:
                st.markdown("<h6> Demo Requests: 2025 vs 2024</h6>", unsafe_allow_html=True)
                demo_time = demo_df.copy()
                demo_monthly = demo_time.groupby(['Year', 'Month']).size().reset_index(name='Count')
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df_chart = pd.DataFrame({
                    "Month": month_order,
                    "This Year": demo_monthly[demo_monthly['Year'] == this_year].set_index("Month").reindex(month_order)["Count"].fillna(0).values,
                    "Last Year": demo_monthly[demo_monthly['Year'] == last_year].set_index("Month").reindex(month_order)["Count"].fillna(0).values
                })
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["This Year"], name="This Year", mode="lines+markers"))
                fig3.add_trace(go.Scatter(x=df_chart["Month"], y=df_chart["Last Year"], name="Last Year", mode="lines+markers"))
                fig3.update_layout(height=150, margin=dict(t=20, b=30), xaxis_title="Month", yaxis_title="Demo Requests")
                st.plotly_chart(fig3, use_container_width=True)

            # Monthly Product Requests Over Time
            with row3_col2:
                st.markdown("<h6>Product Requests: 2025 vs 2024</h6>", unsafe_allow_html=True)
                product_monthly = df_filtered.groupby(['Year', 'Month']).size().reset_index(name='Count')
                df_chart2 = pd.DataFrame({
                    "Month": month_order,
                    "This Year": product_monthly[product_monthly['Year'] == this_year].set_index("Month").reindex(month_order)["Count"].fillna(0).values,
                    "Last Year": product_monthly[product_monthly['Year'] == last_year].set_index("Month").reindex(month_order)["Count"].fillna(0).values
                })
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_chart2["Month"], y=df_chart2["This Year"], name="This Year", mode="lines+markers"))
                fig4.add_trace(go.Scatter(x=df_chart2["Month"], y=df_chart2["Last Year"], name="Last Year", mode="lines+markers"))
                fig4.update_layout(height=150, margin=dict(t=20, b=30), xaxis_title="Month", yaxis_title="Product Requests")
                st.plotly_chart(fig4, use_container_width=True)                       

        # ----AI MODEL VIEW
        elif st.session_state.view == "SESSION OUTCOME PREDICTOR":

            st.markdown("""
                <style>
                    h6 {
                        margin-bottom: 0.2rem;
                    }
                    .element-container:has(h6) + div {
                        margin-top: -10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.markdown("<h4> Session Outcome Predictor </h4>", unsafe_allow_html=True)

            uploaded_file = st.file_uploader("### 📤 Upload Session Data (CSV)", type=["csv"])

            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    input_df = input_df[model_features]

                    # Preprocess and predict
                    X_processed = preprocessor.transform(input_df)
                    predictions = model.predict(X_processed)
                    input_df['Predicted Outcome'] = predictions

                    display_df = input_df[['Predicted Outcome'] + model_features].head(1000000)

                    # Layout with 3 columns: left, spacer, right
                    col1, spacer, col2 = st.columns([4, 0.2, 3])

                    with col1:
                        
                        st.markdown("<h6>📈 Prediction Results</h6>", unsafe_allow_html=True)
                        st.dataframe(display_df, use_container_width=True, height=180)
                        # Download button
                        csv = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="session_outcome_predictions.csv",
                            mime="text/csv"
                        )

                    with col2:                        
                        st.markdown("<h6>📊 Model Performance</h6>", unsafe_allow_html=True)
                        try:
                            report_df = pd.read_csv("classification_report.csv", index_col=0)
                            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True, height=180)
                        except Exception as e:
                            st.error(f"Could not load classification report: {e}")

                except Exception as e:
                    st.error(f"Error processing file: {e}")


        # --- Logout at bottom of sidebar ---
        st.sidebar.markdown("---")
        if st.sidebar.button("🔒 Logout", key="bottom-logout"):
            logout()


if __name__ == "__main__":
    main()



     