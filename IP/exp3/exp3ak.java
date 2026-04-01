import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class exp3ak {
    public static void main(String[] args) {
        try {
            // Load input grayscale image
            File inputFile = new File("exp3.jpg"); // <-- replace with your grayscale image path
            BufferedImage inputImage = ImageIO.read(inputFile);
            int width = inputImage.getWidth();
            int height = inputImage.getHeight();
            int L = 256;  // number of gray levels
            // Create output image (same size first)
            BufferedImage negativeImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            // Apply negative transform
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = inputImage.getRGB(x, y);
                    int gray = pixel & 0xFF; // grayscale value
                   int neg = (L - 1) - gray;
                    int newPixel = (neg << 16) | (neg << 8) | neg;
                    negativeImage.setRGB(x, y, newPixel);
                }
            }
            // Increase output size (e.g., 2x)
            int newWidth = width * 2;
            int newHeight = height * 2;
            BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);
            // Scale using Graphics2D
            Graphics2D g = resizedImage.createGraphics();
            g.drawImage(negativeImage, 0, 0, newWidth, newHeight, null);
            g.dispose();
            // Save the resized negative image
            File outputFile = new File("negative_resized.jpg");
            ImageIO.write(resizedImage, "jpg", outputFile);
            System.out.println("Negative image created and resized successfully!");
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
}
