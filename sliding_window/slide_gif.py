import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def calculate_rgb_brightness(window, target_rgb):
    if window.shape[2] == 4:
        window = window[:, :, :3]
    mean_rgb = np.mean(window, axis=(0, 1))
    brightness = np.sum(np.abs(mean_rgb - target_rgb))
    return mean_rgb, brightness

def create_sliding_window_gif(image_path, output_path, window_height=20, step_size=10, duration=0.1):
    # Load the spectrogram image
    img = Image.open(image_path)

    # Add white padding around the image
    border_width = 50
    img_with_border = ImageOps.expand(img, border=border_width, fill='white')

    img_width, img_height = img.size
    padded_img_width, padded_img_height = img_with_border.size

    # Define target RGB values
    target_rgb = [71.59845682123655, 206.32637970430108, 147.64024529569892]

    # Create a list to hold frames for the GIF
    frames = []

    # Prepare for plot
    y_positions = []
    brightness_scores = []

    # Variable to keep track of the brightest window
    max_brightness = float('-inf')
    brightest_window = None

    for start in range(border_width, border_width + img_height - window_height + 1, step_size):
        # Create a copy of the image to draw the window
        frame_img = img_with_border.copy()
        draw = ImageDraw.Draw(frame_img)
        
        # Draw the sliding window (as a rectangle)
        draw.rectangle([(border_width, start), (border_width + img_width, start + window_height)], outline="magenta", width=3)

        # Calculate the brightness score for the current window
        window = np.array(img_with_border.crop((border_width, start, border_width + img_width, start + window_height)))
        mean_rgb, brightness = calculate_rgb_brightness(window, target_rgb)

        # Draw the brightness score on the image
        draw.text((10, 10), f"Brightness: {brightness:.2f}", fill="white")

        # Convert the frame to RGB
        frame_img = frame_img.convert("RGB")

        # Update plot data
        y_positions.append(start - border_width)
        brightness_scores.append(brightness)

        # Check if this is the brightest window so far
        if brightness > max_brightness:
            max_brightness = brightness
            brightest_window = (border_width, start, border_width + img_width, start + window_height)

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted plot size to be in between
        ax.plot(y_positions, brightness_scores, 'm-')  # Magenta plot line
        ax.set_xlim(0, img_height)
        ax.set_ylim(0, max(brightness_scores) + 100)  # Adjust based on expected brightness range
        ax.set_xlabel('Y-coordinate of Window')
        ax.set_ylabel('Brightness')
        ax.set_title('Brightness vs. Y-Position')
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

    # Add frames to highlight the brightest window
    for _ in range(10):  # Repeat for a few frames to make the pause noticeable
        frame_img = img_with_border.copy()
        draw = ImageDraw.Draw(frame_img)
        draw.rectangle(brightest_window, outline="red", width=3)  # Highlight the brightest window in red
        draw.text((10, 10), f"Brightest: {max_brightness:.2f}", fill="white")
        frame_img = frame_img.convert("RGB")

        # Create plot with brightest window highlighted
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted plot size to be in between
        ax.plot(y_positions, brightness_scores, 'm-')  # Magenta plot line
        ax.set_xlim(0, img_height)
        ax.set_ylim(0, max(brightness_scores) + 100)  # Adjust based on expected brightness range
        ax.set_xlabel('Y-Position of Window')
        ax.set_ylabel('Brightness')
        ax.set_title('Brightness vs. Y-Position')
        ax.axvline(x=brightest_window[1] - border_width, color='red', linestyle='--')  # Highlight the brightest window position
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
    parser = argparse.ArgumentParser(description='Create a GIF of a sliding window scanning across a spectrogram image, showing the brightness score for each window and a plot of brightness vs. y-position.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input spectrogram image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output GIF')
    parser.add_argument('--window_height', type=int, default=20, help='Height of the sliding window (default: 20)')
    parser.add_argument('--step_size', type=int, default=10, help='Number of pixels the window moves in each step (default: 10)')
    parser.add_argument('--duration', type=float, default=0.1, help='Duration of each frame in the GIF in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    create_sliding_window_gif(
        args.image_path,
        args.output_path,
        args.window_height,
        args.step_size,
        args.duration
    )

if __name__ == "__main__":
    main()
