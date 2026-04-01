import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
public class ex4ip {
    // Applies thresholding at a given threshold value
    public static BufferedImage applyThreshold(BufferedImage original, int threshold) {
        int width = original.getWidth();
        int height = original.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = original.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                // Convert to grayscale
                int gray = (r + g + b) / 3;
                // Apply threshold
                int binary = (gray >= threshold) ? 255 : 0;
                int newPixel = (binary << 16) | (binary << 8) | binary;
                result.setRGB(x, y, newPixel);
            }
        }
        return result;
    }
    public static void main(String[] args) {
        try {
            // Load original image (change path as needed)
            File input = new File("ex4.jpg"); // Replace with your image path
            BufferedImage original = ImageIO.read(input);
            // Threshold values
            int[] thresholds = {64, 128, 192};
            for (int threshold : thresholds) {
                BufferedImage thresholdedImage = applyThreshold(original, threshold);
                // Save output image
                File output = new File("threshold_" + threshold + ".png");
                ImageIO.write(thresholdedImage, "png", output);
                System.out.println("Saved thresholded image at threshold " + threshold + " as " + output.getName());
            }
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}

