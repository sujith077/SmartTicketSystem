import streamlit as st
import pandas as pd
import joblib
import json
import os
import time
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(page_title="NCHS IT Ticket System", page_icon="🎫", layout="wide")

# --- DATABASE & SESSION CONFIGURATION ---
DB_FILE = "tickets.json"
SESSION_FILE = "user_session.json"
SESSION_TIMEOUT_SECONDS = 600  # 10 minutes session inactivity timeout

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

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
                if time.time() - data.get("last_active", 0) < SESSION_TIMEOUT_SECONDS:
                    return data.get("user"), data.get("role")
        except:
            return None, None
    return None, None

def save_session(user, role):
    with open(SESSION_FILE, "w") as f:
        json.dump({
            "user": user,
            "role": role,
            "last_active": time.time()
        }, f)

def clear_session():
    if os.path.exists(SESSION_FILE):
        try:
            os.remove(SESSION_FILE)
        except:
            pass

if 'form_generation_id' not in st.session_state:
    st.session_state.form_generation_id = 0

AUTHORIZED_USERS = {
    "itsupport@nchs.edu.lk": "admin@123",
    "sujith.b@nchs.edu.lk": "user@123",
    "ayesha.k@nchs.edu.lk": "user@456"
}

# --- INITIALIZE SESSION FROM STORAGE ---
saved_user, saved_role = load_session()

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = saved_user
if 'user_role' not in st.session_state:
    st.session_state.user_role = saved_role

if st.session_state.logged_in_user:
    save_session(st.session_state.logged_in_user, st.session_state.user_role)

# --- LOGIN GATEWAY ---
if st.session_state.logged_in_user is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🎫 Secure IT Support Gateway")
        st.write("Please log in using your authorized corporate credentials.")
        
        with st.form("login_gateway"):
            username_input = st.text_input("Username / Email Address", placeholder="e.g., sujith.b@nchs.edu.lk")
            password_input = st.text_input("Password", type="password", placeholder="••••••••")
            login_submit = st.form_submit_button("Access Portal", use_container_width=True)
            
            if login_submit:
                clean_user = username_input.strip().lower()
                if clean_user in AUTHORIZED_USERS and AUTHORIZED_USERS[clean_user] == password_input:
                    role = "Admin" if clean_user == "itsupport@nchs.edu.lk" else "User"
                    st.session_state.logged_in_user = clean_user
                    st.session_state.user_role = role
                    save_session(clean_user, role)
                    st.rerun()
                else:
                    st.error("❌ Access Denied: Invalid username or incorrect password.")
    st.stop()

user_email = st.session_state.logged_in_user
is_admin = (st.session_state.user_role == "Admin")

# Sidebar navigation
st.sidebar.markdown(f"### 👤 User Info")
st.sidebar.markdown(f"**Email:** `{user_email}`")
st.sidebar.markdown(f"**Role:** `{st.session_state.user_role}`")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Log Out", use_container_width=True):
    st.session_state.logged_in_user = None
    st.session_state.user_role = None
    clear_session()
    st.rerun()

# --- AUTO-REFRESHING ADMIN DASHBOARD ---
@st.fragment(run_every="5s")
def render_admin_dashboard():
    ticket_database = load_local_database()
    st.session_state.ticket_database = ticket_database

    st.title("👨‍💻 NCHS IT System Administrator Portal")
    st.caption("🔄 **Live Feed:** Dashboard automatically synchronizes new tickets every 5 seconds.")
    st.markdown("---")
    
    if len(ticket_database) > 0:
        admin_df = pd.DataFrame(ticket_database)
        
        # Operational Metrics
        st.subheader("📊 Operational Summary")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1:
            st.metric(label="Total Tickets", value=len(admin_df))
        with kcol2:
            pending_count = len(admin_df[admin_df['Status'] == 'Pending'])
            st.metric(label="⏳ Pending Triage", value=pending_count)
        with kcol3:
            critical_count = len(admin_df[admin_df['Assigned_Priority'] == 'Critical'])
            st.metric(label="🚨 Critical Tickets", value=critical_count)
        with kcol4:
            triage_count = len(admin_df[admin_df['Routing_Status'].str.contains('Manual Triage', na=False)])
            st.metric(label="⚠️ Triage Flags", value=triage_count)
            
        st.markdown("---")
        
        # Quick Action Console
        st.subheader("⚙️ Quick Action Console")
        st.caption("Select a ticket below to update its Status or override its Assigned Priority.")
        
        admin_df['Ticket_ID'] = [i + 1 for i in range(len(admin_df))]
        
        c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
        with c1:
            selected_display_id = st.selectbox("Ticket ID", options=admin_df['Ticket_ID'].tolist())
        
        list_index = int(selected_display_id) - 1
        target_ticket = ticket_database[list_index]
        
        with c2:
            new_status = st.selectbox(
                "Update Status", 
                options=["Pending", "Processing", "Completed"],
                index=["Pending", "Processing", "Completed"].index(target_ticket['Status'])
            )
        with c3:
            new_priority = st.selectbox(
                "Override Priority", 
                options=["Critical", "High", "Medium", "Low"],
                index=["Critical", "High", "Medium", "Low"].index(target_ticket['Assigned_Priority'])
            )
        with c4:
            st.write(" ")
            st.write(" ")
            if st.button("💾 Save Update", use_container_width=True):
                ticket_database[list_index]['Status'] = new_status
                ticket_database[list_index]['Assigned_Priority'] = new_priority
                save_to_local_database(ticket_database)
                st.session_state.ticket_database = ticket_database
                st.toast(f"Ticket #{selected_display_id} updated successfully!", icon="✅")
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Active Operations Queue")
        st.caption("🎨 **Color Key:** 🟡 Yellow = Pending | 🔵 Blue = Processing | 🟢 Green = Completed")

        clean_df = admin_df.copy()
        clean_df.rename(columns={
            "Ticket_ID": "ID",
            "Timestamp": "Date & Time",
            "Title": "Ticket Title",
            "Assigned_Priority": "Priority",
            "Routing_Status": "Action Flag",
            "User_Email": "User"
        }, inplace=True)

        display_cols = ["ID", "Date & Time", "Ticket Title", "Priority", "Status", "Action Flag", "User"]
        clean_df = clean_df[display_cols]

        def highlight_status(row):
            status = row['Status']
            if status == 'Pending':
                return ['background-color: #fff9c4; color: #574500; font-weight: 500;'] * len(row)
            elif status == 'Processing':
                return ['background-color: #e3f2fd; color: #0d47a1; font-weight: 500;'] * len(row)
            elif status == 'Completed':
                return ['background-color: #e8f5e9; color: #1b5e20; font-weight: 500;'] * len(row)
            return [''] * len(row)

        styled_df = clean_df.style.apply(highlight_status, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No tickets have been submitted yet. Awaiting live incoming submissions...")

# --- AUTO-REFRESHING USER TICKET REGISTRY ---
@st.fragment(run_every="5s")
def render_user_tickets():
    st.subheader("📋 My Support Tickets Registry")
    st.caption("🎨 **Color Key:** 🟡 Yellow = Pending | 🔵 Blue = Processing | 🟢 Green = Completed | 🔄 Live Sync Every 5s")
    
    db = load_local_database()
    if len(db) > 0:
        full_df = pd.DataFrame(db)
        user_df = full_df[full_df['User_Email'] == user_email].copy()
        
        if not user_df.empty:
            user_df['ID'] = [i + 1 for i in range(len(user_df))]
            
            rename_map = {
                "Timestamp": "Date & Time",
                "Title": "Ticket Title",
                "Assigned_Priority": "Priority",
                "Routing_Status": "Action Flag"
            }
            user_df.rename(columns=rename_map, inplace=True)

            display_cols = ["ID", "Date & Time", "Ticket Title", "Priority", "Status", "Action Flag"]
            valid_cols = [col for col in display_cols if col in user_df.columns]
            clean_user_df = user_df[valid_cols]

            def highlight_user_status(row):
                status = row.get('Status', '')
                if status == 'Pending':
                    return ['background-color: #fff9c4; color: #574500; font-weight: 500;'] * len(row)
                elif status == 'Processing':
                    return ['background-color: #e3f2fd; color: #0d47a1; font-weight: 500;'] * len(row)
                elif status == 'Completed':
                    return ['background-color: #e8f5e9; color: #1b5e20; font-weight: 500;'] * len(row)
                return [''] * len(row)

            styled_user_df = clean_user_df.style.apply(highlight_user_status, axis=1)

            st.dataframe(
                styled_user_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No tickets logged yet.")
    else:
        st.info("No tickets logged yet.")

if is_admin:
    render_admin_dashboard()

# --- USER PANEL ---
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎫 NCHS IT Support Ticket Portal")
    
    tab1, tab2 = st.tabs(["🆕 Raise New IT Ticket", "📋 View My Submitted Tickets"])
    
    with tab1:
        st.write("Submit your technical issue below with a detailed description for automated classification.")
        
        @st.cache_resource
        def load_pipeline():
            model_path = 'priority_pipeline.pkl'
            
            def train_and_save_pipeline():
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import RandomForestClassifier

                np.random.seed(42)
                departments = ['Academic', 'HR', 'Finance', 'IT', 'Sales', 'Marketing', 'Operations']
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
                return auto_pipeline

            if os.path.exists(model_path):
                try:
                    return joblib.load(model_path)
                except Exception:
                    return train_and_save_pipeline()
            else:
                return train_and_save_pipeline()

        pipeline = load_pipeline()

        with st.form(key=f"prediction_form_{st.session_state.form_generation_id}"):
            ticket_title = st.text_input("Ticket Title / Summary", placeholder="e.g., Unable to access examination portal")
            ticket_desc = st.text_area("Detailed Issue Description", placeholder="Describe what happened, error messages, and impact...")
            
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Originating Department", ['Academic', 'HR', 'Finance', 'IT', 'Sales', 'Marketing', 'Operations'], index=None)
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

                prediction = pipeline.predict(input_data)[0]
                probs = pipeline.predict_proba(input_data)[0]
                confidence = round(float(np.max(probs)) * 100, 1)

                if confidence >= 85:
                    routing_status = "Recommended (Pending Review)"
                elif confidence >= 65:
                    routing_status = "Recommended (Pending Review)"
                else:
                    routing_status = "Flagged for Manual Triage"

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

                current_tickets = load_local_database()
                current_tickets.append(new_ticket_entry)
                save_to_local_database(current_tickets)

                # Persist prediction details across reruns
                st.session_state.last_prediction = {
                    "priority": prediction,
                    "confidence": confidence,
                    "routing": routing_status
                }

                # Reset inputs for next ticket
                st.session_state.form_generation_id += 1
                st.rerun()

        # Display result card after page refresh
        if 'last_prediction' in st.session_state:
            res = st.session_state.last_prediction
            st.markdown(f"### Predicted Priority: **{res['priority']}** *(Confidence: {res['confidence']}%)*")
            st.info(f"**System Routing Action:** {res['routing']}")
            st.success("Ticket successfully logged!")

    with tab2:
        render_user_tickets()
