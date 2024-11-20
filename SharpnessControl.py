import cv2
class SharpnessControl:
    def laplacian_score(image_path):
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform edge detection using the Laplace operator.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Calculate the variance of the Laplace operator.
        variance = laplacian.var()
        # cv2.imshow("Demo", gray)
        # cv2.waitKey(0)
        # Set the threshold
        threshold = 100
        # Determine whether the image is blurry.
        if variance > threshold:
            print("The image is not blurry, Laplacian operator variance:", variance)
        else:
            print("The image is blurry, Laplacian operator variance:", variance)