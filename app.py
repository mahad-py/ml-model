import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LEX — Underwriting Prediction Demo", layout="wide", page_icon="🏦")

CAT_COLS = ['gender','nationality_group','city','employment_status','employer_sector',
            'channel','product_type','risk_band']
NUM_COLS = ['age','months_in_job','monthly_salary_sar','other_monthly_income_sar',
            'total_monthly_income_sar','existing_monthly_obligations_sar','nafath_verified',
            'yakeen_match','requested_amount_or_limit_sar','requested_tenor_months',
            'annual_profit_rate','requested_monthly_payment_est_sar','policy_dbr_cap',
            'dbr_if_requested','max_affordable_new_payment_sar','policy_salary_multiple_cap',
            'max_approvable_limit_sar','risk_score_300_900']
ALL_FEATURES = CAT_COLS + NUM_COLS

# ---------- THEME ----------
st.markdown("""
<style>
.block-container{padding-top:2rem;}
h1,h2,h3{color:#0f2138;}
.metric-card{
    background:#f6f8fb; border:1px solid #e1e6ee; border-radius:10px;
    padding:14px 18px; text-align:center;
}
.flag-ok{background:#e3f5ec; color:#2f9e6e; padding:10px 16px; border-radius:8px; font-weight:600;}
.flag-warn{background:#fdf3e7; color:#d98c2b; padding:10px 16px; border-radius:8px; font-weight:600;}
.flag-bad{background:#fbeae6; color:#c2452f; padding:10px 16px; border-radius:8px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 RADIAN8 LEX — Underwriting Decision Engine")
st.caption("Upload historical applicant data to train the model, then score a new applicant against learned patterns.")

# ---------- SIDEBAR: DATA UPLOAD ----------
st.sidebar.header("1. Training Data")
uploaded = st.sidebar.file_uploader(
    "Upload 1 to 4 applicant dataset files (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True
)
sheet_name = st.sidebar.text_input("Sheet name in each file", value="10K")

@st.cache_data(show_spinner=False)
def load_data(file_names, sheet):
    frames = []
    for f in uploaded:
        try:
            frames.append(pd.read_excel(f, sheet_name=sheet))
        except Exception as e:
            st.sidebar.warning(f"Could not read '{sheet}' from {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df[ALL_FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    y = df['decision']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              random_state=42, verbosity=-1)
    clf.fit(X_train, y_train, categorical_feature=CAT_COLS)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')

    # rule baseline on same test set
    idx = X_test.index
    base = df.loc[idx]
    rule_pred = np.where(
        base['nafath_verified'] == 0, 'declined',
        np.where((base['risk_score_300_900'] >= 560) & (base['max_approvable_limit_sar'] > 0),
                 np.where(base['requested_amount_or_limit_sar'] <= base['max_approvable_limit_sar'],
                          'approved_full', 'approved_reduced'),
                 'declined')
    )
    rule_acc = accuracy_score(y_test, rule_pred)

    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return clf, acc, f1, rule_acc, importance

if uploaded:
    df = load_data([f.name for f in uploaded], sheet_name)
    if df.empty:
        st.error("No data loaded. Check the sheet name matches all uploaded files.")
        st.stop()
    st.sidebar.success(f"Loaded {len(df):,} applicants from {len(uploaded)} file(s)")
    with st.spinner("Training decision model on uploaded data..."):
        clf, acc, f1, rule_acc, importance = train_models(df)
    st.sidebar.metric("Model accuracy (holdout)", f"{acc*100:.2f}%")
    st.sidebar.metric("Policy-rule baseline", f"{rule_acc*100:.2f}%")
    st.sidebar.caption(f"Model lift over hardcoded rule: **{(acc-rule_acc)*100:+.2f}pp**")
else:
    st.info("👈 Upload the historical applicant dataset (.xlsx) in the sidebar to train the model and unlock the demo.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Pattern Analysis", "🧪 Score a New Applicant", "📁 Batch Scoring"])

# ============================================================
# TAB 1: PATTERN ANALYSIS
# ============================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applicants", f"{len(df):,}")
    c2.metric("Approval rate", f"{(df['decision']!='declined').mean()*100:.1f}%")
    c3.metric("Model accuracy", f"{acc*100:.2f}%")
    c4.metric("Exceptions caught", f"{int((acc-rule_acc)*len(df)*0.2):,}")

    colA, colB = st.columns(2)
    with colA:
        fig = px.pie(df, names='decision', title="Decision Distribution", hole=0.5,
                      color_discrete_sequence=['#2e6ed8','#c2452f','#2f9e6e'])
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        nat = df.groupby('nationality_group')['decision'].apply(lambda x: (x != 'declined').mean()*100).reset_index()
        fig2 = px.bar(nat, x='nationality_group', y='decision', title="Approval Rate by Nationality Group",
                       labels={'decision': 'Approval %'}, color_discrete_sequence=['#2e6ed8'])
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        rb = df.groupby('risk_band').size().reset_index(name='count').sort_values('risk_band')
        fig3 = px.bar(rb, x='risk_band', y='count', title="Applicant Volume by Risk Band",
                       color_discrete_sequence=['#5b8def'])
        st.plotly_chart(fig3, use_container_width=True)
    with colD:
        approved = df[df['decision'] != 'declined']
        if 'writeoff_18m' in approved.columns:
            wo = approved.groupby('risk_band')['writeoff_18m'].mean().reset_index()
            wo['writeoff_18m'] *= 100
            fig4 = px.line(wo, x='risk_band', y='writeoff_18m', markers=True,
                            title="Write-off Rate (%) by Risk Band",
                            color_discrete_sequence=['#c2452f'])
            st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Top Decision Drivers")
    imp_df = importance.head(10).reset_index()
    imp_df.columns = ['feature', 'importance']
    fig5 = px.bar(imp_df, x='importance', y='feature', orientation='h',
                   color_discrete_sequence=['#0f2138'])
    fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# TAB 2: SCORE A NEW APPLICANT
# ============================================================
with tab2:
    st.subheader("Enter New Applicant Details")
    st.caption("Fields the applicant/system provides directly. Policy-derived fields (DBR, affordability caps, risk band) are calculated automatically below — exactly as LEX's preprocessing layer would do before this model ever sees the case.")

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", sorted(df['gender'].unique()))
            age = st.number_input("Age", 18, 75, 32)
            nationality_group = st.selectbox("Nationality Group", sorted(df['nationality_group'].unique()))
            city = st.selectbox("City", sorted(df['city'].unique()))
            employment_status = st.selectbox("Employment Status", sorted(df['employment_status'].unique()))
            employer_sector = st.selectbox("Employer Sector", sorted(df['employer_sector'].unique()))
        with c2:
            months_in_job = st.number_input("Months in Job", 0, 480, 24)
            monthly_salary_sar = st.number_input("Monthly Salary (SAR)", 0.0, 200000.0, 12000.0, step=500.0)
            other_monthly_income_sar = st.number_input("Other Monthly Income (SAR)", 0.0, 100000.0, 0.0, step=500.0)
            existing_monthly_obligations_sar = st.number_input("Existing Monthly Obligations (SAR)", 0.0, 100000.0, 2000.0, step=100.0)
            nafath_verified = st.selectbox("Nafath ID Verified", ["Yes", "No"]) == "Yes"
            yakeen_match = st.selectbox("Yakeen Match", ["Yes", "No"]) == "Yes"
        with c3:
            channel = st.selectbox("Channel", sorted(df['channel'].unique()))
            product_type = st.selectbox("Product Type", sorted(df['product_type'].unique()))
            requested_amount_or_limit_sar = st.number_input("Requested Amount (SAR)", 1000.0, 1000000.0, 50000.0, step=1000.0)
            requested_tenor_months = st.number_input("Requested Tenor (months)", 3, 60, 24)
            annual_profit_rate = st.number_input("Annual Profit Rate", 0.05, 0.45, 0.27, step=0.01)
            risk_score_300_900 = st.slider("Risk Score (300-900)", 300, 900, 650)

        submitted = st.form_submit_button("🔍 Predict Decision", use_container_width=True)

    if submitted:
        total_income = monthly_salary_sar + other_monthly_income_sar
        req_payment = (requested_amount_or_limit_sar * (1 + annual_profit_rate*requested_tenor_months/12)) / requested_tenor_months
        policy_dbr_cap = 0.45
        policy_salary_multiple_cap = 15
        dbr_if_requested = (existing_monthly_obligations_sar + req_payment) / total_income if total_income > 0 else 1.0
        max_affordable_payment = max(policy_dbr_cap * total_income - existing_monthly_obligations_sar, 0)
        affordability_limit = max_affordable_payment * requested_tenor_months / (1 + annual_profit_rate*requested_tenor_months/12)
        salary_multiple_limit = policy_salary_multiple_cap * monthly_salary_sar
        max_approvable_limit = min(affordability_limit, salary_multiple_limit)

        def band(s):
            if s >= 700: return 'A'
            if s >= 650: return 'B'
            if s >= 600: return 'C'
            if s >= 550: return 'D'
            return 'E'
        risk_band = band(risk_score_300_900)

        row = pd.DataFrame([{
            'gender': gender, 'age': age, 'nationality_group': nationality_group, 'city': city,
            'employment_status': employment_status, 'employer_sector': employer_sector,
            'months_in_job': months_in_job, 'monthly_salary_sar': monthly_salary_sar,
            'other_monthly_income_sar': other_monthly_income_sar, 'total_monthly_income_sar': total_income,
            'existing_monthly_obligations_sar': existing_monthly_obligations_sar,
            'nafath_verified': int(nafath_verified), 'yakeen_match': int(yakeen_match),
            'channel': channel, 'product_type': product_type,
            'requested_amount_or_limit_sar': requested_amount_or_limit_sar,
            'requested_tenor_months': requested_tenor_months, 'annual_profit_rate': annual_profit_rate,
            'requested_monthly_payment_est_sar': req_payment, 'policy_dbr_cap': policy_dbr_cap,
            'dbr_if_requested': dbr_if_requested, 'max_affordable_new_payment_sar': max_affordable_payment,
            'policy_salary_multiple_cap': policy_salary_multiple_cap, 'max_approvable_limit_sar': max_approvable_limit,
            'risk_score_300_900': risk_score_300_900, 'risk_band': risk_band
        }])

        X_new = row[ALL_FEATURES].copy()
        for c in CAT_COLS:
            X_new[c] = X_new[c].astype('category').cat.set_categories(df[c].astype('category').cat.categories)

        pred = clf.predict(X_new)[0]
        proba = clf.predict_proba(X_new)[0]
        confidence = proba.max() * 100

        # rule baseline for comparison / exception flag
        rule_pred = 'declined'
        if nafath_verified and risk_score_300_900 >= 560 and max_approvable_limit > 0:
            rule_pred = 'approved_full' if requested_amount_or_limit_sar <= max_approvable_limit else 'approved_reduced'

        approved_amount = 0 if pred == 'declined' else min(requested_amount_or_limit_sar, max_approvable_limit)

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            color = {"approved_full": "🟢", "approved_reduced": "🟡", "declined": "🔴"}[pred]
            st.markdown(f"### {color} Decision: **{pred.replace('_',' ').title()}**")
            st.caption(f"Model confidence: {confidence:.1f}%")
        with r2:
            st.markdown(f"### 💰 Approved Amount")
            st.markdown(f"## SAR {approved_amount:,.0f}")
            if pred == 'approved_reduced':
                st.caption(f"Reduced from requested SAR {requested_amount_or_limit_sar:,.0f}")
        with r3:
            st.markdown(f"### 📈 Risk Band: **{risk_band}**")
            st.caption(f"Score: {risk_score_300_900} / 900")

        st.divider()
        st.subheader("Review & Pattern Flags")
        f1c, f2c = st.columns(2)
        with f1c:
            if pred == rule_pred:
                st.markdown(f'<div class="flag-ok">✅ Model agrees with standard policy rule — straightforward case</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-bad">🚩 Exception flag: model decision ({pred}) differs from standard policy rule ({rule_pred}) — recommend human-in-loop review</div>', unsafe_allow_html=True)
        with f2c:
            if not nafath_verified:
                st.markdown(f'<div class="flag-bad">🚫 Identity not verified via Nafath — automatic decline per policy</div>', unsafe_allow_html=True)
            elif not yakeen_match:
                st.markdown(f'<div class="flag-warn">⚠️ Yakeen mismatch — recommend secondary identity check before disbursing</div>', unsafe_allow_html=True)
            elif dbr_if_requested > 0.9:
                st.markdown(f'<div class="flag-warn">⚠️ Debt burden ratio if approved at requested amount is very high ({dbr_if_requested*100:.0f}%) — affordability stretch</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-ok">✅ No identity or affordability red flags detected</div>', unsafe_allow_html=True)

        with st.expander("See calculated policy fields used by the model"):
            st.dataframe(row[['dbr_if_requested','max_affordable_new_payment_sar','max_approvable_limit_sar','risk_band']].T.rename(columns={0: 'value'}))

# ============================================================
# TAB 3: BATCH SCORING
# ============================================================
with tab3:
    st.subheader("Score a Batch of New Applicants")
    st.caption("Upload a CSV/XLSX of new applicants with the same raw columns (no need to pre-calculate policy fields — they're derived automatically).")
    batch_file = st.file_uploader("Upload new applicants file", type=["csv", "xlsx"], key="batch")

    if batch_file is not None:
        if batch_file.name.endswith(".csv"):
            new_df = pd.read_csv(batch_file)
        else:
            new_df = pd.read_excel(batch_file)

        base_required = ['monthly_salary_sar','requested_amount_or_limit_sar','requested_tenor_months',
                          'annual_profit_rate','existing_monthly_obligations_sar','risk_score_300_900']
        missing_base = [c for c in base_required if c not in new_df.columns]
        if missing_base:
            st.error(f"❌ Your uploaded file is missing these required raw columns needed before any calculation: {missing_base}.")
            st.caption(f"Columns found in your file: {list(new_df.columns)}")
            st.stop()

        new_df['total_monthly_income_sar'] = new_df['monthly_salary_sar'] + new_df.get('other_monthly_income_sar', 0)
        new_df['requested_monthly_payment_est_sar'] = (new_df['requested_amount_or_limit_sar'] * (1 + new_df['annual_profit_rate']*new_df['requested_tenor_months']/12)) / new_df['requested_tenor_months']
        new_df['policy_dbr_cap'] = 0.45
        new_df['policy_salary_multiple_cap'] = 15
        new_df['dbr_if_requested'] = (new_df['existing_monthly_obligations_sar'] + new_df['requested_monthly_payment_est_sar']) / new_df['total_monthly_income_sar']
        new_df['max_affordable_new_payment_sar'] = (new_df['policy_dbr_cap']*new_df['total_monthly_income_sar'] - new_df['existing_monthly_obligations_sar']).clip(lower=0)
        afford_limit = new_df['max_affordable_new_payment_sar'] * new_df['requested_tenor_months'] / (1 + new_df['annual_profit_rate']*new_df['requested_tenor_months']/12)
        salary_limit = new_df['policy_salary_multiple_cap'] * new_df['monthly_salary_sar']
        new_df['max_approvable_limit_sar'] = pd.concat([afford_limit, salary_limit], axis=1).min(axis=1)
        new_df['risk_band'] = new_df['risk_score_300_900'].apply(lambda s: 'A' if s>=700 else 'B' if s>=650 else 'C' if s>=600 else 'D' if s>=550 else 'E')

        missing_cols = [c for c in ALL_FEATURES if c not in new_df.columns]
        if missing_cols:
            st.error(f"❌ Your uploaded file is missing these required columns: {missing_cols}. "
                      f"Please check your column headers match the expected schema (see README) and re-upload.")
            st.caption(f"Columns found in your file: {list(new_df.columns)}")
            st.stop()

        X_batch = new_df[ALL_FEATURES].copy()
        for c in CAT_COLS:
            X_batch[c] = X_batch[c].astype('category').cat.set_categories(df[c].astype('category').cat.categories)

        new_df['ML_decision'] = clf.predict(X_batch)
        new_df['confidence_%'] = (clf.predict_proba(X_batch).max(axis=1)*100).round(1)
        new_df['approved_amount_sar'] = np.where(
            new_df['ML_decision']=='declined', 0,
            np.minimum(new_df['requested_amount_or_limit_sar'], new_df['max_approvable_limit_sar'])
        ).round(2)

        st.success(f"Scored {len(new_df)} applicants")
        st.dataframe(new_df[['nationality_group','city','monthly_salary_sar','requested_amount_or_limit_sar',
                              'risk_score_300_900','risk_band','ML_decision','approved_amount_sar','confidence_%']],
                     use_container_width=True)
        st.download_button("⬇️ Download scored results (CSV)", new_df.to_csv(index=False), "scored_applicants.csv")













"""
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LEX — Underwriting Prediction Demo", layout="wide", page_icon="🏦")

CAT_COLS = ['gender','nationality_group','city','employment_status','employer_sector',
            'channel','product_type','risk_band']
NUM_COLS = ['age','months_in_job','monthly_salary_sar','other_monthly_income_sar',
            'total_monthly_income_sar','existing_monthly_obligations_sar','nafath_verified',
            'yakeen_match','requested_amount_or_limit_sar','requested_tenor_months',
            'annual_profit_rate','requested_monthly_payment_est_sar','policy_dbr_cap',
            'dbr_if_requested','max_affordable_new_payment_sar','policy_salary_multiple_cap',
            'max_approvable_limit_sar','risk_score_300_900']
ALL_FEATURES = CAT_COLS + NUM_COLS

# ---------- THEME ----------
st.markdown("""
<style>
.block-container{padding-top:2rem;}
h1,h2,h3{color:#0f2138;}
.metric-card{
    background:#f6f8fb; border:1px solid #e1e6ee; border-radius:10px;
    padding:14px 18px; text-align:center;
}
.flag-ok{background:#e3f5ec; color:#2f9e6e; padding:10px 16px; border-radius:8px; font-weight:600;}
.flag-warn{background:#fdf3e7; color:#d98c2b; padding:10px 16px; border-radius:8px; font-weight:600;}
.flag-bad{background:#fbeae6; color:#c2452f; padding:10px 16px; border-radius:8px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("🏦 RADIAN8 LEX — Underwriting Decision Engine")
st.caption("Upload historical applicant data to train the model, then score a new applicant against learned patterns.")

# ---------- SIDEBAR: DATA UPLOAD ----------
st.sidebar.header("1. Training Data")
uploaded = st.sidebar.file_uploader("Upload historical applicant dataset (.xlsx)", type=["xlsx"])
sheet_name = st.sidebar.text_input("Sheet name", value="10K")

@st.cache_data(show_spinner=False)
def load_data(file, sheet):
    return pd.read_excel(file, sheet_name=sheet)

@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df[ALL_FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    y = df['decision']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              random_state=42, verbosity=-1)
    clf.fit(X_train, y_train, categorical_feature=CAT_COLS)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')

    # rule baseline on same test set
    idx = X_test.index
    base = df.loc[idx]
    rule_pred = np.where(
        base['nafath_verified'] == 0, 'declined',
        np.where((base['risk_score_300_900'] >= 560) & (base['max_approvable_limit_sar'] > 0),
                 np.where(base['requested_amount_or_limit_sar'] <= base['max_approvable_limit_sar'],
                          'approved_full', 'approved_reduced'),
                 'declined')
    )
    rule_acc = accuracy_score(y_test, rule_pred)

    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return clf, acc, f1, rule_acc, importance

if uploaded is not None:
    df = load_data(uploaded, sheet_name)
    st.sidebar.success(f"Loaded {len(df):,} applicants")
    with st.spinner("Training decision model on uploaded data..."):
        clf, acc, f1, rule_acc, importance = train_models(df)
    st.sidebar.metric("Model accuracy (holdout)", f"{acc*100:.2f}%")
    st.sidebar.metric("Policy-rule baseline", f"{rule_acc*100:.2f}%")
    st.sidebar.caption(f"Model lift over hardcoded rule: **{(acc-rule_acc)*100:+.2f}pp**")
else:
    st.info("👈 Upload the historical applicant dataset (.xlsx) in the sidebar to train the model and unlock the demo.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Pattern Analysis", "🧪 Score a New Applicant", "📁 Batch Scoring"])

# ============================================================
# TAB 1: PATTERN ANALYSIS
# ============================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applicants", f"{len(df):,}")
    c2.metric("Approval rate", f"{(df['decision']!='declined').mean()*100:.1f}%")
    c3.metric("Model accuracy", f"{acc*100:.2f}%")
    c4.metric("Exceptions caught", f"{int((acc-rule_acc)*len(df)*0.2):,}")

    colA, colB = st.columns(2)
    with colA:
        fig = px.pie(df, names='decision', title="Decision Distribution", hole=0.5,
                      color_discrete_sequence=['#2e6ed8','#c2452f','#2f9e6e'])
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        nat = df.groupby('nationality_group')['decision'].apply(lambda x: (x != 'declined').mean()*100).reset_index()
        fig2 = px.bar(nat, x='nationality_group', y='decision', title="Approval Rate by Nationality Group",
                       labels={'decision': 'Approval %'}, color_discrete_sequence=['#2e6ed8'])
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        rb = df.groupby('risk_band').size().reset_index(name='count').sort_values('risk_band')
        fig3 = px.bar(rb, x='risk_band', y='count', title="Applicant Volume by Risk Band",
                       color_discrete_sequence=['#5b8def'])
        st.plotly_chart(fig3, use_container_width=True)
    with colD:
        approved = df[df['decision'] != 'declined']
        if 'writeoff_18m' in approved.columns:
            wo = approved.groupby('risk_band')['writeoff_18m'].mean().reset_index()
            wo['writeoff_18m'] *= 100
            fig4 = px.line(wo, x='risk_band', y='writeoff_18m', markers=True,
                            title="Write-off Rate (%) by Risk Band",
                            color_discrete_sequence=['#c2452f'])
            st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Top Decision Drivers")
    imp_df = importance.head(10).reset_index()
    imp_df.columns = ['feature', 'importance']
    fig5 = px.bar(imp_df, x='importance', y='feature', orientation='h',
                   color_discrete_sequence=['#0f2138'])
    fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# TAB 2: SCORE A NEW APPLICANT
# ============================================================
with tab2:
    st.subheader("Enter New Applicant Details")
    st.caption("Fields the applicant/system provides directly. Policy-derived fields (DBR, affordability caps, risk band) are calculated automatically below — exactly as LEX's preprocessing layer would do before this model ever sees the case.")

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", sorted(df['gender'].unique()))
            age = st.number_input("Age", 18, 75, 32)
            nationality_group = st.selectbox("Nationality Group", sorted(df['nationality_group'].unique()))
            city = st.selectbox("City", sorted(df['city'].unique()))
            employment_status = st.selectbox("Employment Status", sorted(df['employment_status'].unique()))
            employer_sector = st.selectbox("Employer Sector", sorted(df['employer_sector'].unique()))
        with c2:
            months_in_job = st.number_input("Months in Job", 0, 480, 24)
            monthly_salary_sar = st.number_input("Monthly Salary (SAR)", 0.0, 200000.0, 12000.0, step=500.0)
            other_monthly_income_sar = st.number_input("Other Monthly Income (SAR)", 0.0, 100000.0, 0.0, step=500.0)
            existing_monthly_obligations_sar = st.number_input("Existing Monthly Obligations (SAR)", 0.0, 100000.0, 2000.0, step=100.0)
            nafath_verified = st.selectbox("Nafath ID Verified", ["Yes", "No"]) == "Yes"
            yakeen_match = st.selectbox("Yakeen Match", ["Yes", "No"]) == "Yes"
        with c3:
            channel = st.selectbox("Channel", sorted(df['channel'].unique()))
            product_type = st.selectbox("Product Type", sorted(df['product_type'].unique()))
            requested_amount_or_limit_sar = st.number_input("Requested Amount (SAR)", 1000.0, 1000000.0, 50000.0, step=1000.0)
            requested_tenor_months = st.number_input("Requested Tenor (months)", 3, 60, 24)
            annual_profit_rate = st.number_input("Annual Profit Rate", 0.05, 0.45, 0.27, step=0.01)
            risk_score_300_900 = st.slider("Risk Score (300-900)", 300, 900, 650)

        submitted = st.form_submit_button("🔍 Predict Decision", use_container_width=True)

    if submitted:
        total_income = monthly_salary_sar + other_monthly_income_sar
        req_payment = (requested_amount_or_limit_sar * (1 + annual_profit_rate*requested_tenor_months/12)) / requested_tenor_months
        policy_dbr_cap = 0.45
        policy_salary_multiple_cap = 15
        dbr_if_requested = (existing_monthly_obligations_sar + req_payment) / total_income if total_income > 0 else 1.0
        max_affordable_payment = max(policy_dbr_cap * total_income - existing_monthly_obligations_sar, 0)
        affordability_limit = max_affordable_payment * requested_tenor_months / (1 + annual_profit_rate*requested_tenor_months/12)
        salary_multiple_limit = policy_salary_multiple_cap * monthly_salary_sar
        max_approvable_limit = min(affordability_limit, salary_multiple_limit)

        def band(s):
            if s >= 700: return 'A'
            if s >= 650: return 'B'
            if s >= 600: return 'C'
            if s >= 550: return 'D'
            return 'E'
        risk_band = band(risk_score_300_900)

        row = pd.DataFrame([{
            'gender': gender, 'age': age, 'nationality_group': nationality_group, 'city': city,
            'employment_status': employment_status, 'employer_sector': employer_sector,
            'months_in_job': months_in_job, 'monthly_salary_sar': monthly_salary_sar,
            'other_monthly_income_sar': other_monthly_income_sar, 'total_monthly_income_sar': total_income,
            'existing_monthly_obligations_sar': existing_monthly_obligations_sar,
            'nafath_verified': int(nafath_verified), 'yakeen_match': int(yakeen_match),
            'channel': channel, 'product_type': product_type,
            'requested_amount_or_limit_sar': requested_amount_or_limit_sar,
            'requested_tenor_months': requested_tenor_months, 'annual_profit_rate': annual_profit_rate,
            'requested_monthly_payment_est_sar': req_payment, 'policy_dbr_cap': policy_dbr_cap,
            'dbr_if_requested': dbr_if_requested, 'max_affordable_new_payment_sar': max_affordable_payment,
            'policy_salary_multiple_cap': policy_salary_multiple_cap, 'max_approvable_limit_sar': max_approvable_limit,
            'risk_score_300_900': risk_score_300_900, 'risk_band': risk_band
        }])

        X_new = row[ALL_FEATURES].copy()
        for c in CAT_COLS:
            X_new[c] = X_new[c].astype('category').cat.set_categories(df[c].astype('category').cat.categories)

        pred = clf.predict(X_new)[0]
        proba = clf.predict_proba(X_new)[0]
        confidence = proba.max() * 100

        # rule baseline for comparison / exception flag
        rule_pred = 'declined'
        if nafath_verified and risk_score_300_900 >= 560 and max_approvable_limit > 0:
            rule_pred = 'approved_full' if requested_amount_or_limit_sar <= max_approvable_limit else 'approved_reduced'

        approved_amount = 0 if pred == 'declined' else min(requested_amount_or_limit_sar, max_approvable_limit)

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            color = {"approved_full": "🟢", "approved_reduced": "🟡", "declined": "🔴"}[pred]
            st.markdown(f"### {color} Decision: **{pred.replace('_',' ').title()}**")
            st.caption(f"Model confidence: {confidence:.1f}%")
        with r2:
            st.markdown(f"### 💰 Approved Amount")
            st.markdown(f"## SAR {approved_amount:,.0f}")
            if pred == 'approved_reduced':
                st.caption(f"Reduced from requested SAR {requested_amount_or_limit_sar:,.0f}")
        with r3:
            st.markdown(f"### 📈 Risk Band: **{risk_band}**")
            st.caption(f"Score: {risk_score_300_900} / 900")

        st.divider()
        st.subheader("Review & Pattern Flags")
        f1c, f2c = st.columns(2)
        with f1c:
            if pred == rule_pred:
                st.markdown(f'<div class="flag-ok">✅ Model agrees with standard policy rule — straightforward case</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-bad">🚩 Exception flag: model decision ({pred}) differs from standard policy rule ({rule_pred}) — recommend human-in-loop review</div>', unsafe_allow_html=True)
        with f2c:
            if not nafath_verified:
                st.markdown(f'<div class="flag-bad">🚫 Identity not verified via Nafath — automatic decline per policy</div>', unsafe_allow_html=True)
            elif not yakeen_match:
                st.markdown(f'<div class="flag-warn">⚠️ Yakeen mismatch — recommend secondary identity check before disbursing</div>', unsafe_allow_html=True)
            elif dbr_if_requested > 0.9:
                st.markdown(f'<div class="flag-warn">⚠️ Debt burden ratio if approved at requested amount is very high ({dbr_if_requested*100:.0f}%) — affordability stretch</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-ok">✅ No identity or affordability red flags detected</div>', unsafe_allow_html=True)

        with st.expander("See calculated policy fields used by the model"):
            st.dataframe(row[['dbr_if_requested','max_affordable_new_payment_sar','max_approvable_limit_sar','risk_band']].T.rename(columns={0: 'value'}))

# ============================================================
# TAB 3: BATCH SCORING
# ============================================================
with tab3:
    st.subheader("Score a Batch of New Applicants")
    st.caption("Upload a CSV/XLSX of new applicants with the same raw columns (no need to pre-calculate policy fields — they're derived automatically).")
    batch_file = st.file_uploader("Upload new applicants file", type=["csv", "xlsx"], key="batch")

    if batch_file is not None:
        if batch_file.name.endswith(".csv"):
            new_df = pd.read_csv(batch_file)
        else:
            new_df = pd.read_excel(batch_file)

        base_required = ['monthly_salary_sar','requested_amount_or_limit_sar','requested_tenor_months',
                          'annual_profit_rate','existing_monthly_obligations_sar','risk_score_300_900']
        missing_base = [c for c in base_required if c not in new_df.columns]
        if missing_base:
            st.error(f"❌ Your uploaded file is missing these required raw columns needed before any calculation: {missing_base}.")
            st.caption(f"Columns found in your file: {list(new_df.columns)}")
            st.stop()

        new_df['total_monthly_income_sar'] = new_df['monthly_salary_sar'] + new_df.get('other_monthly_income_sar', 0)
        new_df['requested_monthly_payment_est_sar'] = (new_df['requested_amount_or_limit_sar'] * (1 + new_df['annual_profit_rate']*new_df['requested_tenor_months']/12)) / new_df['requested_tenor_months']
        new_df['policy_dbr_cap'] = 0.45
        new_df['policy_salary_multiple_cap'] = 15
        new_df['dbr_if_requested'] = (new_df['existing_monthly_obligations_sar'] + new_df['requested_monthly_payment_est_sar']) / new_df['total_monthly_income_sar']
        new_df['max_affordable_new_payment_sar'] = (new_df['policy_dbr_cap']*new_df['total_monthly_income_sar'] - new_df['existing_monthly_obligations_sar']).clip(lower=0)
        afford_limit = new_df['max_affordable_new_payment_sar'] * new_df['requested_tenor_months'] / (1 + new_df['annual_profit_rate']*new_df['requested_tenor_months']/12)
        salary_limit = new_df['policy_salary_multiple_cap'] * new_df['monthly_salary_sar']
        new_df['max_approvable_limit_sar'] = pd.concat([afford_limit, salary_limit], axis=1).min(axis=1)
        new_df['risk_band'] = new_df['risk_score_300_900'].apply(lambda s: 'A' if s>=700 else 'B' if s>=650 else 'C' if s>=600 else 'D' if s>=550 else 'E')

        missing_cols = [c for c in ALL_FEATURES if c not in new_df.columns]
        if missing_cols:
            st.error(f"❌ Your uploaded file is missing these required columns: {missing_cols}. "
                      f"Please check your column headers match the expected schema (see README) and re-upload.")
            st.caption(f"Columns found in your file: {list(new_df.columns)}")
            st.stop()

        X_batch = new_df[ALL_FEATURES].copy()
        for c in CAT_COLS:
            X_batch[c] = X_batch[c].astype('category').cat.set_categories(df[c].astype('category').cat.categories)

        new_df['ML_decision'] = clf.predict(X_batch)
        new_df['confidence_%'] = (clf.predict_proba(X_batch).max(axis=1)*100).round(1)
        new_df['approved_amount_sar'] = np.where(
            new_df['ML_decision']=='declined', 0,
            np.minimum(new_df['requested_amount_or_limit_sar'], new_df['max_approvable_limit_sar'])
        ).round(2)

        st.success(f"Scored {len(new_df)} applicants")
        st.dataframe(new_df[['nationality_group','city','monthly_salary_sar','requested_amount_or_limit_sar',
                              'risk_score_300_900','risk_band','ML_decision','approved_amount_sar','confidence_%']],
                     use_container_width=True)
        st.download_button("⬇️ Download scored results (CSV)", new_df.to_csv(index=False), "scored_applicants.csv")"""
