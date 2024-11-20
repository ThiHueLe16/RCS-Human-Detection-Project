from BrightnessControl import BrightnessControl
from ContrastAdjustment import ContrastAdjustment
from ObjectDetection import ObjectDetection
from PictureData import PictureData
from SharpnessControl import SharpnessControl
from CircleBoundaryMonitor import CircleBoundaryMonitor
import cv2

def main():
    """
    Main method to run the PPM brightness adjustment for P6 format.
    """
    print("P6 PPM Adjustment")
    print("===================================================================================================")
    #
    # # 1.variance: Get user inputs
    # input_path = input("Enter the path to the input P6 PPM file (e.g., input.ppm): ").strip()
    # output_path = input("Enter the path to save the output P6 PPM file (e.g., output.ppm): ").strip()
    # try:
    #     brightness_offset = int(input("Enter the brightness offset (e.g., 50 to brighten, -50 to darken): ").strip())
    # except ValueError:
    #     print("Invalid brightness offset. Please enter an integer.")
    #     return
    #
    # # Perform brightness adjustment
    # try:
    #     BrightnessControl.adjust_brightness_ppm_p6(input_path, output_path, brightness_offset)
    #     print(f"Brightness adjustment complete. Output saved to {output_path}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # 2.Variance: use defined input
    input_path="/Users/thihuele/HumanDetection/pythonProject/checkimage.ppm"
    output_path= "./output.ppm"
    brightness=10
    contrast=65


    pictureData = PictureData(input_path)
    pictureData.display_info()

    colorAnalysis = BrightnessControl.analyze_color_brightness(pictureData)

    print("Color Brightness Analysis:")
    print(f"Average Brightness: {colorAnalysis['average_brightness']:.2f}")
    print(f"R Mean: {colorAnalysis['R_mean']:.2f}")
    print(f"G Mean: {colorAnalysis['G_mean']:.2f}")
    print(f"B Mean: {colorAnalysis['B_mean']:.2f}")
    print(f"Brightness Status: {colorAnalysis['status']}")

    BrightnessControl.adjust_brightness_ppm_p6(pictureData,brightness)
    ContrastAdjustment.adjust_contrast(pictureData,contrast )
    pictureData.write_ppm("./output.ppm")



    # colorAnalysis = BrightnessControl.analyze_color_brightness(PictureData("./output.ppm"))
    #
    # print("Color Brightness Analysis:")
    # print(f"Average Brightness: {colorAnalysis['average_brightness']:.2f}")
    # print(f"R Mean: {colorAnalysis['R_mean']:.2f}")
    # print(f"G Mean: {colorAnalysis['G_mean']:.2f}")
    # print(f"B Mean: {colorAnalysis['B_mean']:.2f}")
    # print(f"Brightness Status: {colorAnalysis['status']}")
    #
    # print("Sharpness Analysis:")
    # blurry_image_path = "/Users/thihuele/Downloads/50611477431_8ea50523cf_c.jpg"
    # SharpnessControl.laplacian_score(blurry_image_path)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

#     test object in circleboudary




    # cam = cv2.VideoCapture(0)  # Open default camera
    # if not cam.isOpened():
    #     print("Error: Could not open camera.")
    #     return
    # # Read the first frame to determine frame dimensions
    # ret, frame = cam.read()
    # if not ret:
    #     print("Error: Could not read frame.")
    #     return
    # # Calculate the center of the frame
    # height, width, _ = frame.shape
    # center = (width // 2, height // 2)
    # # Circle boundaries
    # outer_radius = min(width, height) // 3  # Example: Outer radius is 1/3 of the smaller dimension
    # inner_radius = outer_radius // 2  # Example: Inner radius is half of outer radius
    #
    # # Initialize the boundary monitor
    # monitor = CircleBoundaryMonitor(center, outer_radius, inner_radius)
    #
    # while True:
    #     ret, frame = cam.read()
    #     if not ret:
    #         break
    #
    #     # Draw the circles (boundaries)
    #     cv2.circle(frame, center, outer_radius, (0, 255, 255), 2)  # Outer circle
    #     cv2.circle(frame, center, inner_radius, (0, 0, 255), 2)    # Inner circle
    #
    #     # Detect objects and check zones
    #     frame = ObjectDetection.detect_objects(frame, center, monitor)
    #
    #     # Show the frame
    #     cv2.imshow("Boundary Monitor", frame)
    #
    #     # Exit on pressing 'q'
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cam.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()