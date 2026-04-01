import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class exp8i {
    @SuppressWarnings({"UseSpecificCatch", "CallToPrintStackTrace"})
    public static void main(String[] args) {
        try {
            // Load the input image
            BufferedImage input = ImageIO.read(new File("ak.png"));
            int width = input.getWidth();
            int height = input.getHeight();

            // Create images for grayscale, erosion, and dilation results
            BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            BufferedImage eroded = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            BufferedImage dilated = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

            // Convert input image to grayscale
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int rgb = input.getRGB(x, y);
                    int r = (rgb >> 16) & 0xff;
                    int g = (rgb >> 8) & 0xff;
                    int b = rgb & 0xff;
                    int grayVal = (r + g + b) / 3;
                    int newPixel = (grayVal << 16) | (grayVal << 8) | grayVal;
                    gray.setRGB(x, y, newPixel);
                }
            }

            // Perform Erosion
            for (int x = 1; x < width - 1; x++) {
                for (int y = 1; y < height - 1; y++) {
                    int min = 255;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            int pixel = gray.getRGB(x + i, y + j) & 0xff;
                            if (pixel < min) min = pixel;
                        }
                    }
                    int newPixel = (min << 16) | (min << 8) | min;
                    eroded.setRGB(x, y, newPixel);
                }
            }

            // Perform Dilation
            for (int x = 1; x < width - 1; x++) {
                for (int y = 1; y < height - 1; y++) {
                    int max = 0;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            int pixel = gray.getRGB(x + i, y + j) & 0xff;
                            if (pixel > max) max = pixel;
                        }
                    }
                    int newPixel = (max << 16) | (max << 8) | max;
                    dilated.setRGB(x, y, newPixel);
                }
            }

            // Save output images
            ImageIO.write(eroded, "jpg", new File("eroded_output.jpg"));
            ImageIO.write(dilated, "jpg", new File("dilated_output.jpg"));

            System.out.println("Erosion and Dilation completed successfully!");
            System.out.println("Output saved as eroded_output.jpg and dilated_output.jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
