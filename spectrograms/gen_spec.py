import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
mat_file = scipy.io.loadmat('00014.mat')

# Extract the 'full_data' key
full_data = mat_file['full_data']

# Extract the main data array
data_array = full_data[0, 0][0]

# Print the shape of the data array to understand its dimensions
print("Data array shape:", data_array.shape)

# Check if the data needs reshaping or additional processing
# If the data array is not in a 2D format, you may need to reshape it here
# For this example, we assume it's already in the correct format

# Generate the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(data_array, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram')

# Save the spectrogram to an image file
output_image_path = 'spectrogram_output.png'
plt.savefig(output_image_path)


print(f"Spectrogram saved to {output_image_path}")
