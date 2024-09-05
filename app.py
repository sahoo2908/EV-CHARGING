from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io
import base64
from fpdf import FPDF
import os

app = Flask(__name__)

# Load the CSV data
df = pd.read_csv('Bhubaneswar_Charging_Station.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_nearest_nsga', methods=['POST'])
def find_nearest_nsga():
    user_lat = request.json['lat']
    user_lon = request.json['lng']
    user_coords = (user_lat, user_lon)
    
    nearest_station = find_nearest_station(user_coords, df)
    
    plot_url, plot_path = plot_map(user_coords, (nearest_station['Latitude'], nearest_station['Longitude']), df)
    
    response = {
        'station': nearest_station.to_dict(),
        'plot_url': plot_url
    }
    
    return jsonify(response)

@app.route('/find_nearest_power_nsga', methods=['POST'])
def find_nearest_power_nsga():
    user_lat = request.json['lat']
    user_lon = request.json['lng']
    range_km = request.json['range']
    user_coords = (user_lat, user_lon)
    
    highest_power_station = find_nearest_station_with_highest_power(user_coords, df, range_km)
    
    plot_url, plot_path = plot_map(user_coords, (highest_power_station['Latitude'], highest_power_station['Longitude']), df, highest_power_station)
    
    response = {
        'station': highest_power_station.to_dict(),
        'plot_url': plot_url
    }
    
    return jsonify(response)

@app.route('/find_nearest_dijkstra', methods=['POST'])
def find_nearest_dijkstra():
    user_lat = request.json['lat']
    user_lon = request.json['lng']
    user_coords = (user_lat, user_lon)
    
    nearest_station = find_nearest_station(user_coords, df)
    
    plot_url, plot_path = plot_map(user_coords, (nearest_station['Latitude'], nearest_station['Longitude']), df)
    
    response = {
        'station': nearest_station.to_dict(),
        'plot_url': plot_url
    }
    
    return jsonify(response)

@app.route('/find_nearest_power_dijkstra', methods=['POST'])
def find_nearest_power_dijkstra():
    user_lat = request.json['lat']
    user_lon = request.json['lng']
    range_km = request.json['range']
    user_coords = (user_lat, user_lon)
    
    highest_power_station = find_nearest_station_with_highest_power(user_coords, df, range_km)
    
    plot_url, plot_path = plot_map(user_coords, (highest_power_station['Latitude'], highest_power_station['Longitude']), df, highest_power_station)
    
    response = {
        'station': highest_power_station.to_dict(),
        'plot_url': plot_url
    }
    
    return jsonify(response)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="EV Charging Station Finder Result", ln=True, align='C')
    
    # Ensure 'static' directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
        
    if os.path.exists("static/result_plot.png"):
        pdf.image("static/result_plot.png", x=10, y=20, w=180)
    
    response = io.BytesIO()
    pdf.output(response, 'F')
    response.seek(0)
    
    return send_file(response, as_attachment=True, download_name="result.pdf", mimetype='application/pdf')

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def find_nearest_station(user_coords, df):
    distances = []
    for _, row in df.iterrows():
        station_coords = (row['Latitude'], row['Longitude'])
        dist = euclidean_distance(user_coords, station_coords)
        distances.append((dist, row))
    
    distances.sort(key=lambda x: x[0])
    return distances[0][1]

def find_nearest_station_with_highest_power(user_coords, df, range_km):
    filtered_df = df[df.apply(lambda row: euclidean_distance(user_coords, (row['Latitude'], row['Longitude'])) <= range_km, axis=1)]
    if filtered_df.empty:
        return None
    
    highest_power_station = filtered_df.loc[filtered_df['Power (W)'].idxmax()]
    return highest_power_station

def plot_map(user_coords, station_coords, df, highest_power_station=None):
    fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size

    ax.plot(user_coords[1], user_coords[0], 'ro', markersize=10, label='User Location')
    
    if highest_power_station is not None:
        ax.plot(highest_power_station['Longitude'], highest_power_station['Latitude'], 'go', markersize=10, label='Highest Power Charging Station')
    else:
        ax.plot(station_coords[1], station_coords[0], 'go', markersize=10, label='Nearest Charging Station')
    
    ax.plot(df['Longitude'], df['Latitude'], 'bo', markersize=5, label='Charging Stations')
    
    if highest_power_station is not None:
        line = Line2D([user_coords[1], highest_power_station['Longitude']], [user_coords[0], highest_power_station['Latitude']], color='black', linestyle='--')
    else:
        line = Line2D([user_coords[1], station_coords[1]], [user_coords[0], station_coords[0]], color='black', linestyle='--')
    
    ax.add_line(line)
    
    ax.text(user_coords[1], user_coords[0], 'User', fontsize=12, ha='right')
    if highest_power_station is not None:
        ax.text(highest_power_station['Longitude'], highest_power_station['Latitude'], 'Highest Power Station', fontsize=12, ha='left')
    else:
        ax.text(station_coords[1], station_coords[0], 'Nearest Station', fontsize=12, ha='left')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Charging Stations and Shortest Path')
    ax.legend()
    
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Save the plot to a file for PDF download
    if not os.path.exists("static"):
        os.makedirs("static")
        
    with open("static/result_plot.png", "wb") as f:
        f.write(base64.b64decode(plot_url))
    
    return f"data:image/png;base64,{plot_url}", "static/result_plot.png"

if __name__ == '__main__':
    app.run(debug=True)
