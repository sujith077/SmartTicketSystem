import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Generate Synthetic Dataset with Text Descriptions
np.random.seed(42)
n_samples = 2500

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
for _ in range(n_samples):
    cat = np.random.choice(categories)
    title = np.random.choice(sample_titles[cat])
    dept = np.random.choice(departments)
    dev = np.random.choice(devices)
    loc = np.random.choice(locations)
    users = np.random.randint(1, 50)
    crit = np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
    desc = f"{title} reported in {dept} department at {loc} office affecting {users} users."

    data.append({
        'Ticket_Title': title,
        'Ticket_Description': desc,
        'Department': dept,
        'Issue_Category': cat,
        'Device_Type': dev,
        'Affected_Users': users,
        'Business_Critical': crit,
        'Office_Location': loc
    })

df = pd.DataFrame(data)

# Target Assignment Logic
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

# 2. Pipeline Construction (Preprocessing + Classifier)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)

# 3. Export Trained Pipeline
joblib.dump(pipeline, 'priority_pipeline.pkl')
print("✅ Trained pipeline successfully exported to 'priority_pipeline.pkl'")
