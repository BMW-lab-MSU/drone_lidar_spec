import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def calculate_grayscale_error(window, target_grayscale):
    if window.shape[2] == 4:
        window = window[:, :, :3]
    mean_rgb = np.mean(window, axis=(0, 1))
    mean_grayscale = rgb_to_grayscale(mean_rgb)
    grayscale_error = abs(mean_grayscale - target_grayscale)
    return mean_grayscale, grayscale_error

def add_axes_labels(image, img_width, img_height, border_width):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.array(image))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_xticklabels(['0', '0.125', '0.25'])
    ax.set_yticks([0, img_height // 4, img_height // 2, 3 * img_height // 4, img_height])
    ax.set_yticklabels(['1952', '1464', '976', '488', '0'])  # Reversed order
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    fig.tight_layout(pad=3.0)  # Add padding to ensure labels are not cut off
    canvas = FigureCanvas(fig)
    canvas.draw()
    labeled_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    labeled_img = labeled_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    labeled_img = Image.fromarray(labeled_img)
    plt.close(fig)
    return labeled_img

def create_sliding_window_gif(image_path, output_path, window_height=20, step_size=10, duration=0.1):
    # Load the spectrogram image
    img = Image.open(image_path)

    # Add white padding around the image
    border_width = 50
    img_with_border = ImageOps.expand(img, border=border_width, fill='white')

    img_width, img_height = img.size
    padded_img_width, padded_img_height = img_with_border.size

    # Add time and frequency labels to the spectrogram
    labeled_img_with_border = add_axes_labels(img, img_width, img_height, border_width)

    # Define target RGB values and convert to grayscale
    target_rgb = [146.2918103158602, 205.90643800403225, 72.32910836693549]
    target_grayscale = rgb_to_grayscale(target_rgb)

    # Create a list to hold frames for the GIF
    frames = []

    # Prepare for plot
    y_positions = []
    grayscale_error_scores = []

    # Variable to keep track of the window with the lowest error
    min_grayscale_error = float('inf')
    best_window = None

    for start in range(0, img_height - window_height + 1, step_size):
        # Create a copy of the image to draw the window
        frame_img = labeled_img_with_border.copy()
        draw = ImageDraw.Draw(frame_img)
        
        # Draw the sliding window (as a rectangle)
        draw.rectangle([(border_width + 82, start + border_width), (border_width + img_width + 92, start + window_height + border_width)], outline="magenta", width=3)

        # Calculate the grayscale error score for the current window
        window = np.array(img.crop((0, start, img_width, start + window_height)))
        mean_grayscale, grayscale_error = calculate_grayscale_error(window, target_grayscale)

        # Draw the grayscale error score on the image
        #draw.text((border_width + 60, border_width + 10), f"Total Grayscale Error: {grayscale_error:.2f}", fill="white")  # Adjusted text position

        # Convert the frame to RGB
        frame_img = frame_img.convert("RGB")

        # Update plot data
        y_positions.append(start)
        grayscale_error_scores.append(grayscale_error)

        # Check if this is the window with the lowest error so far
        if grayscale_error < min_grayscale_error:
            min_grayscale_error = grayscale_error
            best_window = (border_width + 82, start + border_width, border_width + img_width + 92, start + window_height + border_width)

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted plot size to be in between
        ax.plot(y_positions, grayscale_error_scores, 'm-')  # Magenta plot line
        ax.set_xlim(0, img_height)
        ax.set_ylim(0, max(grayscale_error_scores) + 10)  # Adjust based on expected grayscale error range
        ax.set_xlabel('Y-coordinate of Window')
        ax.set_ylabel('Total Grayscale Error')
        ax.set_title('Total Grayscale Error vs. Y-Position')
        fig.tight_layout(pad=3.0)  # Add padding to ensure labels are not cut off
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = Image.fromarray(plot_img)

        # Combine spectrogram and plot images side by side with padding
        total_width = frame_img.width + plot_img.width + border_width
        combined_img = Image.new('RGB', (total_width, frame_img.height), (255, 255, 255))
        combined_img.paste(frame_img, (0, 0))
        combined_img.paste(plot_img, (frame_img.width + border_width, (frame_img.height - plot_img.height) // 2))

        # Add combined image to frames
        frames.append(combined_img)

        plt.close(fig)  # Close the figure to prevent memory leaks

    # Add frames to highlight the window with the lowest error
    for _ in range(10):  # Repeat for a few frames to make the pause noticeable
        frame_img = labeled_img_with_border.copy()
        draw = ImageDraw.Draw(frame_img)
        draw.rectangle(best_window, outline="red", width=3)  # Highlight the best window in red
        #draw.text((border_width + 60, border_width + 10), f"Best: {min_grayscale_error:.2f}", fill="white")  # Adjusted text position
        frame_img = frame_img.convert("RGB")

        # Create plot with the best window highlighted
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted plot size to be in between
        ax.plot(y_positions, grayscale_error_scores, 'm-')  # Magenta plot line
        ax.set_xlim(0, img_height)
        ax.set_ylim(0, max(grayscale_error_scores) + 10)  # Adjust based on expected grayscale error range
        ax.set_xlabel('Y-Position of Window')
        ax.set_ylabel('Total Grayscale Error')
        ax.set_title('Total Grayscale Error vs. Y-Position')
        ax.axvline(x=best_window[1] - border_width, color='red', linestyle='--')  # Highlight the best window position
        fig.tight_layout(pad=3.0)  # Add padding to ensure labels are not cut off
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = Image.fromarray(plot_img)

        # Combine spectrogram and plot images side by side with padding
        combined_img = Image.new('RGB', (total_width, frame_img.height), (255, 255, 255))
        combined_img.paste(frame_img, (0, 0))
        combined_img.paste(plot_img, (frame_img.width + border_width, (frame_img.height - plot_img.height) // 2))

        # Add combined image to frames
        frames.append(combined_img)

        plt.close(fig)  # Close the figure to prevent memory leaks

    # Save the frames as a GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration * 1000, loop=0)

def main():
    parser = argparse.ArgumentParser(description='Create a GIF of a sliding window scanning across a spectrogram image, showing the grayscale error score for each window and a plot of grayscale error vs. y-position.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input spectrogram image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output GIF')
    parser.add_argument('--window_height', type=int, default=20, help='Height of the sliding window (default: 20)')
    parser.add_argument('--step_size', type=int, default=10,
