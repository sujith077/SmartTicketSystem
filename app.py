import streamlit as st
import pandas as pd
import joblib
import json
import os
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
    
# --- SECURE CREDENTIALS REGISTRY ---
AUTHORIZED_USERS = {
    "itsupport@nchs.edu.lk": "admin@123",    # Admin Account
    "sujith.b@nchs.edu.lk": "user@123",      # staff Account 1
    "ayesha.k@nchs.edu.lk": "user@456"     # staff Account 2
}

# --- CUSTOM FORM LOGIN GATEWAY ---
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

# --- MAIN APPLICATION INTERFACE ---
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

# ----------------- ADMIN INTERFACE -----------------
if is_admin:
    st.write("Welcome to the control console. Below is the live queue where you can review issues and change tracking statuses.")
    st.markdown("---")
    
    if len(st.session_state.ticket_database) > 0:
        admin_df = pd.DataFrame(st.session_state.ticket_database)
        
        # Performance Analytics Dashboard
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
        st.subheader("📋 Active Operations Data Queue & Status Control")
        st.info("💡 Pro Tip: Click directly inside the **Status** column drop-downs below to update tickets in real time.")
        
        # Interactive data editor allowing the admin to dynamically select new status options
        edited_df = st.data_editor(
            admin_df,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    help="Modify the active lifecycle stage of the ticket",
                    options=["Pending", "Processing", "Completed"],
                    required=True,
                )
            },
            disabled=[col for col in admin_df.columns if col != "Status"],
            use_container_width=True
        )
        
        # Automatically update the backend file system if the admin updates a row status
        if not edited_df.equals(admin_df):
            st.session_state.ticket_database = edited_df.to_dict('records')
            save_to_local_database(st.session_state.ticket_database)
            st.toast("System updated successfully!", icon="💾")
            st.rerun()
            
        csv_download_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Ticket Prioritization List (CSV)",
            data=csv_download_data,
            file_name="master_it_prioritization_list.csv",
            mime="text/csv"
        )
    else:
        st.info("No active tickets logged in the current system queue.")

# ----------------- USER INTERFACE -----------------
else:
    # Splitting user workspace into structural layout sections via native tabs
    tab1, tab2 = st.tabs(["🆕 Raise New IT Ticket", "📋 View My Submitted Tickets"])
    
    with tab1:
        st.write("Submit your technical issue below. Our machine learning system will automatically categorize your request.")
        
        @st.cache_resource
        def load_ml_assets():
            model = joblib.load('priority_model.pkl')
            encoders = joblib.load('encoders.pkl')
            return model, encoders

        try:
            model, encoders = load_ml_assets()
        except:
            st.error("Error: Serialized deployment assets were not detected.")
            st.stop()

        with st.form(key=f"prediction_form_{st.session_state.form_generation_id}"):
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Originating Department", encoders['Department'].classes_, index=None, placeholder="Choose Department...")
                category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_, index=None, placeholder="Choose Category...")
                device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_, index=None, placeholder="Choose Device...")
            with col2:
                branch = st.selectbox("Office Branch Location", encoders['Office_Location'].classes_, index=None, placeholder="Choose Branch...")
                affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
                
                impact_choice = st.radio(
                    "How badly is this issue affecting your work?",
                    ["🔴 I cannot work", "🟢 I can still work"],
                    index=None
                )
                
            st.write("") 
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                submit = st.form_submit_button("Compute System Priority Target", use_container_width=True)

        if submit:
            if not department or not category or not device or not branch or not impact_choice:
                st.error("⚠️ Submission Rejected: Please select a value for all fields.")
            else:
                business_critical = "Yes" if "I cannot work" in impact_choice else "No"
                
                # --- PATHWAY A: HYBRID LOGIC / PROGRAMMATIC SAFETY OVERRIDE ---
                if category == "Security" and device == "Server":
                    prediction = "Critical"
                
                # --- PATHWAY B: OPTIMIZED ML INFERENCE PATHWAY ---
                else:
                    # 1. Calculate the feature-engineered term on-the-fly
                    impact_multiplier = 1.5 if business_critical == "Yes" else 0.5
                    calculated_impact_score = affected_users * impact_multiplier

                    # 2. Build the input array utilizing label encoders
                    input_df = pd.DataFrame([{
                        'Department': encoders['Department'].transform([department])[0],
                        'Issue_Category': encoders['Issue_Category'].transform([category])[0],
                        'Device_Type': encoders['Device_Type'].transform([device])[0],
                        'Affected_Users': affected_users,
                        'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0],
                        'Office_Location': encoders['Office_Location'].transform([branch])[0],
                        'Impact_Score': calculated_impact_score
                    }])

                    # 3. Restructure columns to exactly match model training layout
                    column_order = [
                        'Department', 
                        'Issue_Category', 
                        'Device_Type', 
                        'Affected_Users', 
                        'Business_Critical', 
                        'Office_Location', 
                        'Impact_Score'
                    ]
                    input_df = input_df[column_order]

                    # 4. Run Inference
                    prediction = model.predict(input_df)[0]
                
                design_map = {
                    "Critical": {"emoji": "🔴", "color": "#ff4b4b"},
                    "High": {"emoji": "🟠", "color": "#ffa500"},
                    "Medium": {"emoji": "🟡", "color": "#f1c40f"},
                    "Low": {"emoji": "🟢", "color": "#2ecc71"}
                }
                meta = design_map.get(prediction, {"emoji": "ℹ️", "color": "#333333"})
                
                st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{meta['color']}; color:white; font-size:24px; font-weight:bold; text-align:center;'>Resulting Priority Level: {meta['emoji']} {prediction}</div>", unsafe_allow_html=True)
                
                new_ticket_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User_Email": user_email,
                    "Branch_Location": branch,
                    "Department": department,
                    "Issue_Category": category,
                    "Device_Type": device,
                    "Affected_Users": affected_users,
                    "Business_Critical": business_critical,
                    "Assigned_Priority": prediction,
                    "Status": "Pending"  # Enforces requirement: new tickets start as 'Pending'
                }
                
                st.session_state.ticket_database.append(new_ticket_entry)
                save_to_local_database(st.session_state.ticket_database)
                
                st.toast(f"Ticket successfully logged!", icon="✅")
                st.session_state.form_generation_id += 1
                st.button("File Another Support Ticket")
                
    with tab2:
        st.subheader("📋 My Support Tickets Registry")
        if len(st.session_state.ticket_database) > 0:
            full_df = pd.DataFrame(st.session_state.ticket_database)
            # Filter the view so standard users only see rows tied to their own email address
            user_df = full_df[full_df['User_Email'] == user_email]
            
            if not user_df.empty:
                st.dataframe(user_df, use_container_width=True, hide_index=True)
            else:
                st.info("You haven't submitted any IT tickets during this operating period.")
        else:
            st.info("You haven't submitted any IT tickets during this operating period.")
