class ContrastAdjustment:
    #  kontrastärmeres (k = 0) bzw. kontrastreicheres Bild (|k| = 255) :
    @staticmethod
    def adjust_contrast(pictureData, contrast):
        # Compute mean and standard deviation for each color channel
        red_values,  green_values,  blue_values = [], [], []

        for i in range(0, len(pictureData.pixels), 3):
            red_values.append(pictureData.pixels[i])
            green_values.append(pictureData.pixels[i + 1])
            blue_values.append(pictureData.pixels[i + 2])

        def mean(values):
            return sum(values) / len(values)

        def std_deviatiom(values, mean_value):
            return (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5

        # Clamp the values to [0, 255]
        def clamp(value):
            return max(0, min(255, int(value)))

        # Calculate mean and standard deviation for each channel red, green, blue
        red_mean = mean(red_values)
        green_mean = mean(green_values)
        blue_mean = mean(blue_values)

        r_std_dev = std_deviatiom(red_values, red_mean)
        g_std_dev = std_deviatiom(green_values, green_mean)
        b_std_dev = std_deviatiom(blue_values, blue_mean)


        # Adjust contrast
        for i in range(0, len(pictureData.pixels), 3):
            red =pictureData.pixels[i]
            green = pictureData.pixels[i + 1]
            blue = pictureData.pixels[i + 2]

            # Apply contrast adjustment for each channel
            pictureData.pixels[i] =clamp((contrast / r_std_dev) * red + (1 - contrast / r_std_dev) * red_mean)
            pictureData.pixels[i + 1] = clamp((contrast/ g_std_dev) * green  + (1 - contrast / g_std_dev) * green_mean)
            pictureData.pixels[i + 2] =clamp((contrast / b_std_dev) * blue + (1 - contrast / b_std_dev) * blue_mean)

