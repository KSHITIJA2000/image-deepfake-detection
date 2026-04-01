import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class exp2ak {

    public static void main(String[] args) {
        try {
            // Load the colored image
            File input = new File("gray.jpg");  // Replace with your input image path
            BufferedImage coloredImage = ImageIO.read(input);

            // Create a grayscale image of the same size
            BufferedImage grayscaleImage = new BufferedImage(
                    coloredImage.getWidth(),
                    coloredImage.getHeight(),
                    BufferedImage.TYPE_BYTE_GRAY);

            // Convert each pixel to grayscale
            for (int y = 0; y < coloredImage.getHeight(); y++) {
                for (int x = 0; x < coloredImage.getWidth(); x++) {
                    Color c = new Color(coloredImage.getRGB(x, y));
                    int red = c.getRed();
                    int green = c.getGreen();
                    int blue = c.getBlue();

                    // Calculate luminance (grayscale value)
                    int gray = (int)(0.299 * red + 0.587 * green + 0.114 * blue);

                    // Create new grayscale color
                    Color grayColor = new Color(gray, gray, gray);

                    grayscaleImage.setRGB(x, y, grayColor.getRGB());
                }
            }

            // Save the grayscale image
            File output = new File("output_grayscale.jpg");  // Output path
            ImageIO.write(grayscaleImage, "jpg", output);

            System.out.println("Grayscale image created successfully.");

        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
