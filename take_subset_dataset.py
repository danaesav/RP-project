import pandas as pd


def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        # Read the single line containing the IDs
        content = file.read()

        # Remove any trailing newline characters and split by comma
        id_list = content.strip().split(',')

    return id_list


if __name__ == '__main__':
    #Full dataset
    dataset = pd.read_hdf("data/metr-la.h5")
    #Subset sensor ids (separated by coma)
    id_list = read_ids_to_array("data/sensor_graph/graph_sensor_ids_subset.txt")

    filtered_dataset = dataset[id_list]
    #Save subset dataset
    filtered_dataset.to_hdf('data/metr-la_filtered.h5', key='subregion_test', mode='w')
    # print(filtered_dataset)

