import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np
from datetime import datetime

st.set_page_config(page_title="NCHS IT Ticket System", page_icon="🎫", layout="centered")

# --- DATABASE INTEGRATION ---
DB_FILE = "tickets.json"

def load_local_database():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_local_database(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

if 'ticket_database' not in st.session_state:
    st.session_state.ticket_database = load_local_database()

if 'form_generation_id' not in st.session_state:
    st.session_state.form_generation_id = 0

AUTHORIZED_USERS = {
    "itsupport@nchs.edu.lk": "admin@123",
    "sujith.b@nchs.edu.lk": "user@123",
    "ayesha.k@nchs.edu.lk": "user@456"
}

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if st.session_state.logged_in_user is None:
    st.title("🎫 Secure IT Support Gateway")
    st.write("Please log in using your authorized corporate credentials.")
    
    with st.form("login_gateway"):
        username_input = st.text_input("Username / Email Address", placeholder="e.g., sujith.b@nchs.edu.lk")
        password_input = st.text_input("Password", type="password", placeholder="••••••••")
        login_submit = st.form_submit_button("Access Portal")
        
        if login_submit:
            clean_user = username_input.strip().lower()
            if clean_user in AUTHORIZED_USERS and AUTHORIZED_USERS[clean_user] == password_input:
                st.session_state.logged_in_user = clean_user
                st.session_state.user_role = "Admin" if clean_user == "itsupport@nchs.edu.lk" else "User"
                st.rerun()
            else:
                st.error("❌ Access Denied: Invalid username or incorrect password.")
    st.stop()

user_email = st.session_state.logged_in_user
is_admin = (st.session_state.user_role == "Admin")

if is_admin:
    st.markdown("<h1 style='text-align: center;'>👨‍💻 NCHS IT System Administrator Portal</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center;'>🎫 NCHS IT Support Ticket Priority Predictor</h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Logged in as:**\n`{user_email}`")
st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
if st.sidebar.button("Log Out"):
    st.session_state.logged_in_user = None
    st.session_state.user_role = None
    st.rerun()

# --- ADMIN PANEL ---
if is_admin:
    st.write("Welcome to the control console. Below is the live queue where you can review issues and change tracking statuses.")
    st.markdown("---")
    
    if len(st.session_state.ticket_database) > 0:
        admin_df = pd.DataFrame(st.session_state.ticket_database)
        
        st.subheader("📊 System Performance Metrics & Charts")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.metric(label="Total Tickets Logged", value=len(admin_df))
        with kpi_col2:
            critical_count = len(admin_df[admin_df['Assigned_Priority'] == 'Critical'])
            st.metric(label="🚨 Critical Escalations", value=critical_count)
        with kpi_col3:
            avg_impact = round(admin_df['Affected_Users'].mean(), 1)
            st.metric(label="👥 Avg. Impact Radius", value=f"{avg_impact} Users")
            
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Tickets Grouped by Priority**")
            priority_counts = admin_df['Assigned_Priority'].value_counts()
            order = ['Critical', 'High', 'Medium', 'Low']
            priority_counts = priority_counts.reindex([p for p in order if p in priority_counts.index])
            st.bar_chart(priority_counts)
        with chart_col2:
            st.markdown("**Ticket Breakdown by Branch Location**")
            branch_counts = admin_df['Branch_Location'].value_counts()
            st.bar_chart(branch_counts)
            
        st.markdown("---")
        st.subheader("📋 Active Operations Data Queue")
        
        edited_df = st.data_editor(
            admin_df,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Pending", "Processing", "Completed"],
                    required=True,
                )
            },
            disabled=[col for col in admin_df.columns if col != "Status"],
            use_container_width=True
        )
        
        if not edited_df.equals(admin_df):
            st.session_state.ticket_database = edited_df.to_dict('records')
            save_to_local_database(st.session_state.ticket_database)
            st.toast("System updated successfully!", icon="💾")
            st.rerun()

# --- USER PANEL ---
else:
    tab1, tab2 = st.tabs(["🆕 Raise New IT Ticket", "📋 View My Submitted Tickets"])
    
    with tab1:
        st.write("Submit your technical issue below with a detailed description for automated NLP classification.")
        
        @st.cache_resource
        def load_pipeline():
            model_path = 'priority_pipeline.pkl'
            if not os.path.exists(model_path):
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import RandomForestClassifier

                np.random.seed(42)
                departments = ['HR', 'Finance', 'IT', 'Sales', 'Marketing', 'Operations']
                categories = ['Software', 'Hardware', 'Network', 'Access/Login', 'Security']
                devices = ['Laptop', 'Desktop', 'Mobile', 'Printer', 'Server', 'None']
                locations = ['Kandy', 'Colombo']
                sample_titles = {
                    'Security': ['Server firewall failure', 'Phishing email reported', 'Unauthorized access attempt'],
                    'Network': ['VPN connection dropping', 'Wi-Fi slow in Colombo branch', 'Switch port down'],
                    'Software': ['ERP application crash', 'Excel formula error', 'Software license expired'],
                    'Hardware': ['Monitor flickering', 'Printer jam in accounting', 'Laptop battery dying'],
                    'Access/Login': ['Password reset required', 'Account locked out', 'MFA token not received']
                }
                data = []
                for _ in range(2500):
                    cat = np.random.choice(categories)
                    title = np.random.choice(sample_titles[cat])
                    dept = np.random.choice(departments)
                    dev = np.random.choice(devices)
                    loc = np.random.choice(locations)
                    users = np.random.randint(1, 50)
                    crit = np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
                    desc = f"{title} reported in {dept} department at {loc} office affecting {users} users."
                    data.append({
                        'Ticket_Title': title, 'Ticket_Description': desc, 'Department': dept,
                        'Issue_Category': cat, 'Device_Type': dev, 'Affected_Users': users,
                        'Business_Critical': crit, 'Office_Location': loc
                    })
                df = pd.DataFrame(data)
                def assign_priority(row):
                    if row['Issue_Category'] == 'Security' and row['Device_Type'] == 'Server':
                        return 'Critical' if (row['Affected_Users'] > 10 or row['Business_Critical'] == 'Yes') else 'High'
                    score = 0
                    if row['Business_Critical'] == 'Yes': score += 4
                    if row['Affected_Users'] > 20: score += 3
                    elif row['Affected_Users'] > 5: score += 1
                    if row['Issue_Category'] in ['Network', 'Security']: score += 2
                    score += np.random.randint(-1, 2)
                    if score >= 6: return 'Critical'
                    elif score >= 4: return 'High'
                    elif score >= 2: return 'Medium'
                    else: return 'Low'
                df['Priority'] = df.apply(assign_priority, axis=1)
                df['Impact_Score'] = df['Affected_Users'] * df['Business_Critical'].map({'Yes': 1.5, 'No': 0.5})

                categorical_cols = ['Department', 'Issue_Category', 'Device_Type', 'Business_Critical', 'Office_Location']
                numerical_cols = ['Affected_Users', 'Impact_Score']
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('text', TfidfVectorizer(max_features=50, stop_words='english'), 'Ticket_Description'),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                        ('num', StandardScaler(), numerical_cols)
                    ]
                )
                X = df[['Ticket_Description', 'Department', 'Issue_Category', 'Device_Type', 'Business_Critical', 'Office_Location', 'Affected_Users', 'Impact_Score']]
                y = df['Priority']
                auto_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42))
                ])
                auto_pipeline.fit(X, y)
                joblib.dump(auto_pipeline, model_path)

            return joblib.load(model_path)

        pipeline = load_pipeline()

        with st.form(key=f"prediction_form_{st.session_state.form_generation_id}"):
            ticket_title = st.text_input("Ticket Title / Summary", placeholder="e.g., Unable to access examination portal")
            ticket_desc = st.text_area("Detailed Issue Description", placeholder="Describe what happened, error messages, and impact...")
            
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Originating Department", ['HR', 'Finance', 'IT', 'Sales', 'Marketing', 'Operations'], index=None)
                category = st.selectbox("Functional Issue Category", ['Software', 'Hardware', 'Network', 'Access/Login', 'Security'], index=None)
                device = st.selectbox("Primary Device Classification", ['Laptop', 'Desktop', 'Mobile', 'Printer', 'Server', 'None'], index=None)
            with col2:
                branch = st.selectbox("Office Branch Location", ['Kandy', 'Colombo'], index=None)
                affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
                impact_choice = st.radio("Work Impact", ["🔴 I cannot work", "🟢 I can still work"], index=None)
                
            submit = st.form_submit_button("Compute System Priority Target", use_container_width=True)

        if submit:
            if not ticket_title or not ticket_desc or not department or not category or not device or not branch or not impact_choice:
                st.error("⚠️ Submission Rejected: Please fill in all fields.")
            else:
                business_critical = "Yes" if "I cannot work" in impact_choice else "No"
                impact_score = affected_users * (1.5 if business_critical == "Yes" else 0.5)

                input_data = pd.DataFrame([{
                    'Ticket_Description': f"{ticket_title}. {ticket_desc}",
                    'Department': department,
                    'Issue_Category': category,
                    'Device_Type': device,
                    'Business_Critical': business_critical,
                    'Office_Location': branch,
                    'Affected_Users': affected_users,
                    'Impact_Score': impact_score
                }])

                # Run ML Prediction
                prediction = pipeline.predict(input_data)[0]
                probs = pipeline.predict_proba(input_data)[0]
                confidence = round(float(np.max(probs)) * 100, 1)

                # Confidence Routing Logic
                if confidence >= 85:
                    routing_status = "Auto-Assigned"
                elif confidence >= 65:
                    routing_status = "Recommended (Pending Review)"
                else:
                    routing_status = "Flagged for Manual Triage"

                st.markdown(f"### Predicted Priority: **{prediction}** *(Confidence: {confidence}%)*")
                st.info(f"**System Routing Action:** {routing_status}")

                new_ticket_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User_Email": user_email,
                    "Title": ticket_title,
                    "Description": ticket_desc,
                    "Branch_Location": branch,
                    "Department": department,
                    "Issue_Category": category,
                    "Device_Type": device,
                    "Affected_Users": affected_users,
                    "Business_Critical": business_critical,
                    "Assigned_Priority": prediction,
                    "Confidence": f"{confidence}%",
                    "Routing_Status": routing_status,
                    "Status": "Pending"
                }

                st.session_state.ticket_database.append(new_ticket_entry)
                save_to_local_database(st.session_state.ticket_database)
                st.success("Ticket successfully logged!")

    with tab2:
        st.subheader("📋 My Support Tickets Registry")
        if len(st.session_state.ticket_database) > 0:
            full_df = pd.DataFrame(st.session_state.ticket_database)
            user_df = full_df[full_df['User_Email'] == user_email]
            if not user_df.empty:
                st.dataframe(user_df, use_container_width=True, hide_index=True)
            else:
                st.info("No tickets logged yet.")
        else:
            st.info("No tickets logged yet.")
