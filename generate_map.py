import folium
import pandas as pd

# Function to read IDs from file and convert to numpy array
def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        # Read the single line containing the IDs
        content = file.read()

        # Remove any trailing newline characters and split by comma
        id_list = content.strip().split(',')

        # Convert list of strings to integer if IDs are expected to be integers
        id_list = [int(id) for id in id_list]

    return id_list

if __name__ == '__main__':

    sensor_locations = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv")
    sensor_locations_bay = pd.read_csv("data/sensor_graph/graph_sensor_locations_bay.csv")
    # sensor_ids = read_ids_to_array("data/sensor_graph/graph_sensor_ids.txt")
    sensor_ids = []

    # Create a map centered around Los Angeles
    la_map = folium.Map(location=[35.5522, -120.5437], zoom_start=7)
    # la_map = folium.Map(zoom_start=10)
    # Add points for each sensor
    for idx, row in sensor_locations.iterrows():
        if(int(row['sensor_id']) not in sensor_ids):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,  # Set the size of the marker
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"Sensor ID: {int(row['sensor_id'])}"  # Popup text
            ).add_to(la_map)
        else:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,  # Set the size of the marker
                color='yellow',
                fill=True,
                fill_color='yellow',
                fill_opacity=0.6,
                popup=f"Sensor ID: {int(row['sensor_id'])}"  # Popup text
            ).add_to(la_map)

    for idx, row in sensor_locations_bay.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,  # Set the size of the marker
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"Sensor ID: {int(row['sensor_id'])}"  # Popup text
        ).add_to(la_map)

    # Display the map
    la_map.save('LA_AND_BAY_traffic_sensors_map.html')
