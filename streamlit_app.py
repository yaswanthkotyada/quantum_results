import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load Quantum Model Predictions
try:
    data = pd.read_csv('quantum_results.csv')
except FileNotFoundError:
    st.error("quantum_results.csv not found. Make sure to run the quantum script first.")
    st.stop()  # Stop execution if file not found

# Streamlit Dashboard
st.title('Quantum Neural Network for Network Anomaly Detection')
st.write("This application demonstrates a Quantum Neural Network model for detecting network anomalies.")

# Confusion Matrix Visualization
st.subheader("Confusion Matrix")
cm = confusion_matrix(data['Actual'], data['Predicted'])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(fig)

# Distribution of Predictions (Pie Chart)
st.subheader("Prediction Distribution")
pred_counts = data['Predicted'].value_counts()
fig, ax = plt.subplots()
ax.pie(pred_counts, labels=['Normal', 'Attack'], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
st.pyplot(fig)

# Actual vs Predicted Bar Chart
st.subheader("Actual vs Predicted Counts")
actual_counts = data['Actual'].value_counts().rename(index={0: "Normal", 1: "Attack"})
predicted_counts = data['Predicted'].value_counts().rename(index={0: "Normal", 1: "Attack"})

fig, ax = plt.subplots()
df_compare = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts})
df_compare.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
plt.xlabel("Class")
plt.ylabel("Count")
st.pyplot(fig)

# Feature Importance (If Available)
try:
    import joblib
    model = joblib.load('random_forest_model.pkl')  # Load trained model
    feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)

    st.subheader("Feature Importance (RandomForest)")
    fig, ax = plt.subplots()
    feature_importances.nlargest(10).plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)
except:
    st.warning("Feature importance not available. Ensure the model is saved.")

# Show sample predictions
st.subheader("Sample Predictions")
st.write(data.head(10))


