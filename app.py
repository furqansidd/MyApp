from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
import pandas as pd

app = Flask(__name__)

# Sample Dataset
data = {
    "StudyHours": [10, 8, 5, 7, 4, 6, 9, 2, 3, 12],
    "Attendance": [90, 85, 60, 70, 40, 65, 88, 30, 50, 95]
}
df = pd.DataFrame(data)

# Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])

        # Predict cluster
        user_data = [[study_hours, attendance]]
        cluster = kmeans.predict(user_data)[0]

        return jsonify({'cluster': int(cluster)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)



