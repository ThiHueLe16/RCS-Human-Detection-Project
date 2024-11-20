class PictureData:
    def __init__(self, file_path=None):

        self.width = 0
        self.height = 0
        self.max_value = 255
        self.pixels = []

        if file_path:
            self.read_ppm(file_path)

    def read_ppm(self, file_path):
        """
        Reads a PPM file (P6 format) and store its attribute
        """
        with open(file_path, 'rb') as file:
            magic_number = file.readline().strip()
            if magic_number != b"P6":
                raise ValueError("Unsupported PPM format. Only P6 is supported.")

            #  skipping comments
            dimensions = file.readline().strip()
            while dimensions.startswith(b'#'):
                dimensions = file.readline().strip()
            # Read dimensions and max value
            self.width, self.height = map(int, dimensions.split())
            self.max_value = int(file.readline().strip())

            # Read pixel data
            pixel_data = file.read()
            if len(pixel_data) != self.width * self.height * 3:
                raise ValueError("Mismatch between pixel data and image dimensions.")
            self.pixels = list(pixel_data)

    def write_ppm(self, file_path: object) -> object:
        """
        Writes the current picture data to a PPM file (P6 format).
        """
        with open(file_path, 'wb') as f:
            # Write the header
            f.write(b"P6\n")
            f.write(f"{self.width} {self.height}\n".encode())
            f.write(f"{self.max_value}\n".encode())

            # Write pixel data
            f.write(bytes(self.pixels))

    def display_info(self):
        print(f"Image Width: {self.width}")
        print(f"Image Height: {self.height}")
        print(f"Maximum Color Value: {self.max_value}")