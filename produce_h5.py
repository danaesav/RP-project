import h5py
import pandas as pd
import numpy as np
# Load your DataFrame
speed_matrix_df = pd.read_pickle('/Users/macbook/PycharmProjects/RP-project/data/seatlle.pkl')

print(speed_matrix_df)
# Path for the new HDF5 file
output_hdf5_path = '/Users/macbook/PycharmProjects/RP-project/data/seatlle.h5'

# Create the HDF5 file
with h5py.File(output_hdf5_path, 'w') as h5file:
    h5file.create_dataset("df", data=speed_matrix_df )

print(f'HDF5 file created at {output_hdf5_path}')