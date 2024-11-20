import numpy as np

from PictureData import PictureData


class BrightnessControl:
    @staticmethod

    def adjust_brightness_ppm_p6(pictureData, brightness):
        """
        Adjust the brightness of a PPM image in P6 (binary) format.

        Parameters:
        - pictureData: file with picture attribute to adjust the brightness
        - output_path: Path to save the brightness-adjusted PPM file.
        - brightness_offset: Value to add to each RGB component (positive or negative).
        """

        # Read input image
        width, height, max_value, pixels = pictureData.width, pictureData.height,pictureData.max_value, pictureData.pixels

        # Adjust brightness

        for i in range(len(pixels)):
            pixels[i] =  max(0, min(max_value, pixels[i] + brightness))  # Clamp to [0, max_value]

    @staticmethod
    def analyze_color_brightness(pictureData:PictureData):
        """
        Analyze the brightness of an image and classify it as dark, bright, or normal.
        """

        pixels = pictureData.pixels

        # Separate channels
        red_values, green_values, blue_values = [], [], []

        for i in range(0, len(pictureData.pixels), 3):
            red_values.append(pictureData.pixels[i])
            green_values.append(pictureData.pixels[i + 1])
            blue_values.append(pictureData.pixels[i + 2])

        def mean(values):
            return sum(values) / len(values)


        # Clamp the values to [0, 255]
        def clamp(value):
            return max(0, min(255, int(value)))

        # Calculate mean and standard deviation for each channel red, green, blue
        red_mean = mean(red_values)
        green_mean = mean(green_values)
        blue_mean = mean(blue_values)

        # Calculate average brightness
        avg_brightness = (red_mean + green_mean + blue_mean) / 3

        # Define thresholds for classification
        if avg_brightness < 50:
            brightness_status = "Too Dark"
        elif avg_brightness > 200:
            brightness_status = "Too Bright"
        else:
            brightness_status = "Normal Brightness"

        return {
            "average_brightness": avg_brightness,
            "R_mean": red_mean,
            "G_mean": green_mean,
            "B_mean": blue_mean,
            "status": brightness_status
        }



