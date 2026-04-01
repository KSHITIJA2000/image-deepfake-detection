import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class lighttodark {

    public static void main(String[] args) {
        try {
            int L = 256;           // Number of gray levels (0 to L-1)
            int width = L;         // Width = L pixels
            int height = 300;      // Fixed height

            // ✅ Missing semicolon fixed here
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

            for (int i = 0; i < L; i++) {
                // Reverse grayscale intensity: start from white (255) and go to black (0)
                int grayValue = L - 1 - i;

                // Scale down if L > 256
                if (L > 256) {
                    grayValue = (int) (((L - 1 - i) * 255.0) / (L - 1));
                }

                // Create grayscale color
                Color gray = new Color(grayValue, grayValue, grayValue);

                // Fill the entire column with this color
                for (int y = 0; y < height; y++) {
                    image.setRGB(i, y, gray.getRGB());
                }
            }

            // Save the image to file
            File output = new File("light_to_dark_experiment.png");
            ImageIO.write(image, "png", output);

            System.out.println("✅ Light-to-dark gradient image created successfully!");
        } catch (IOException e) {
        }
    }
}
