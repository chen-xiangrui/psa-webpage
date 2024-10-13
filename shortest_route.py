import streamlit as st
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
import networkx as nx

edges = [
    # Asia-Middle East-Europe Routes
    'China-Turkey', 'Turkey-Belgium', 'China-Saudi Arabia', 'Saudi Arabia-Turkey',
    'China-South Korea', 'South Korea-Japan', 'Japan-Turkey', 'China-Singapore',
    'Singapore-Turkey', 'Singapore-Belgium', 'India-Saudi Arabia', 'India-Turkey',
    'Indonesia-Singapore', 'Singapore-Saudi Arabia', 'Vietnam-Singapore', 'Vietnam-Japan',
    'Japan-Italy', 'South Korea-Italy', 'Turkey-Poland', 'Belgium-Poland', 'Italy-Portugal',
    
    # Americas Routes
    'USA-Canada', 'USA-Panama', 'Canada-Panama', 'Panama-Argentina', 'Argentina-USA',
    'Argentina-Brazil', 'USA-Colombia', 'Colombia-Panama', 'Canada-Colombia', 'Panama-Brazil',
    'Argentina-Belgium',
    
    # Asia-Middle East-Europe-Americas Cross Routes
    'Singapore-Argentina', 'Saudi Arabia-Panama', 'Turkey-USA', 'Belgium-USA',
    'Italy-Canada', 'Turkey-Canada', 'Belgium-Argentina', 'Portugal-Panama',
    
    # South-East Asia and Pacific Routes
    'India-Singapore', 'India-Indonesia', 'Indonesia-Thailand', 'Thailand-Singapore',
    'Singapore-Vietnam', 'Thailand-Vietnam',
    
    # Middle East-Europe Connections
    'Saudi Arabia-Italy', 'Saudi Arabia-Belgium', 'Turkey-Italy', 'Turkey-Portugal',
    'Italy-Belgium', 'Portugal-Belgium',
    
    # Americas-Asia-Pacific Connections
    'Panama-Japan', 'Panama-South Korea', 'Canada-Singapore', 'Panama-China',
    
    # South America-Asia Connections
    'Brazil-Singapore', 'Argentina-China', 'Brazil-Italy', 'Brazil-Japan',
    
    # Additional Strategic Routes
    'Colombia-China', 'Colombia-Singapore'
]

coordinates = [
    [37.556, 53.312], [38.114, 11.176], [20.918, 63.322], [35.270, 31.267], [34.884, 123.81], 
    [36.809, 132.512], [9.177, 63.358], [7.775, 109.432], [9.177, 63.358], [35.603, 16.388],
    [23.818, 63.978], [26.830, 34.964], [0.253, 104.701], [13.354, 69.988], [8.372, 107.524],
    [24.518, 125.183], [23.939, 37.072], [23.939, 37.082], [43.957, 32.618], [38.001, 7.224],
    [36.259, -3.865],

    [48.395, -127.321], [24.496, -88.073], [25.823, -77.033], [-19.777, -35.129], [17.705, -60.426],
    [-30.901, -49.921], [14.629, -75.832], [11.277, -79.290], [14.629, -75.832], [4.329, -47.636],
    [37.261, -23.983],

    [-26.588, -134.814], [22.601, -41.510], [22.601, -41.510], [39.337, -46.429], [38.242, -37.676],
    [51.041, -39.513], [-0.079, -28.075], [24.319, -49.856],

    [10.867, 88.752], [7.623, 88.941], [6.061, 105.023], [6.061, 105.028], [8.584, 110.239],
    [8.920, 106.801],

    [33.427, 26.246], [36.011, -3.828], [35.834, 25.173], [37.301, 6.241], [40.244, -10.497], 
    [49.177, -3.556],

    [26.626, -163.756], [23.728, -167.240], [3.135, -172.719], [22.610, 172.775],

    [-36.847, -150.429], [-6.739, -160.938], [25.319, -25.537], [-1.984, -164.784],

    [16.316, -151.422], [-5.028, -158.453]
]

data = {
    'edge': edges,
    'temperature': [],
    'wind_speed': [],
    'wind_direction': [],
    'weather_code': []
}

def get_weather(latitude, longitude):
    # Open-Meteo API URL
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"

    # Send request to the API
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        current_weather = data['current_weather']

        # Extract weather data
        weather_data = [
            current_weather['temperature'], 
            current_weather['windspeed'], 
            current_weather['winddirection'], 
            current_weather['weathercode']
        ]

        return weather_data
    else:
        return "Error: Unable to fetch weather data."

# Input data to run in model
for coordinate in coordinates:
    weather_data = get_weather(coordinate[0], coordinate[1])

    data['temperature'].append(weather_data[0])
    data['wind_speed'].append(weather_data[1])
    data['wind_direction'].append(weather_data[2])
    data['weather_code'].append(weather_data[3])

# Run model and output predicted traffic data
# Load the saved model
model = joblib.load('tabnet_model_3.pkl')

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first 10 rows to check
print(df.head(10))

# Encode the edge (route) as categorical features
edge_encoder = LabelEncoder()
df['edge'] = edge_encoder.fit_transform(df['edge'])

# Separate features and target variable (traffic)
X = df[['edge', 'temperature', 'wind_speed', 'wind_direction', 'weather_code']].values

y_results = model.predict(X)

distances = [
    7438, 3151, 5974, 1494, 730,
    419, 8491, 1653, 5785, 8293,
    2509, 3973, 1627, 4321, 646,
    2228, 8913, 8706, 3963, 970,
    1999,

    6473, 3979, 2723, 5764, 7722,
    1313, 4169, 477, 2469, 4516,
    6625,

    9356, 4321, 9941, 8637, 3887,
    4125, 6625, 4365,

    1925, 3511, 2280, 831, 646,
    681,

    1916, 4002, 1036, 2237, 1999,
    939,

    8169, 8381, 9267, 9131,

    8935, 10578, 5867, 11488,

    9321, 10872
]

# Create a graph
G = nx.Graph()

# Add edges with weights (traffic + distance)
for i, edge in enumerate(edges):
    # Split the edge into nodes
    node1, node2 = edge.split('-')
    
    # Calculate weight as traffic + distance
    traffic_value = max(y_results[i], 0)
    distance = distances[i]
    weight = 40 * traffic_value + distance

    # Add edge to the graph
    G.add_edge(node1, node2, weight=weight)

# Step 3: Shortest Path Algorithm

def shortest_path(start_node, end_node):
    try:
        # Get the shortest path using Dijkstra's algorithm
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return "No path found between the specified nodes."
    except nx.NodeNotFound:
        return "One of the nodes does not exist in the graph."

# Function to get the shortest path
def get_shortest_path(start_node, end_node):
    path = shortest_path(start_node, end_node)
    return path

# Streamlit UI
st.title("Shipping Route Optimizer")
st.write("Enter the start and end locations to find the shortest path.")

# Input fields for start and end locations
start_location = st.text_input("Start Location", "Singapore")
end_location = st.text_input("End Location", "Belgium")

if st.button("Calculate Shortest Path"):
    if start_location and end_location:
        # Call the function to get the shortest path
        shortest_path_result = get_shortest_path(start_location, end_location)

        # Display the result
        st.write(f"The shortest path from {start_location} to {end_location} is:")
        st.write(shortest_path_result)
    else:
        st.warning("Please enter both start and end locations.")

# If you want to show more details or visualizations, you can add more components here.


