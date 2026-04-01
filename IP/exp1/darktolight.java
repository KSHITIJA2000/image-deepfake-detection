import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class darktolight {

    public static void main(String[] args) {
        try {
            int L = 256;           // Number of gray levels (0 to L-1)
            int width = L;         // Width = L pixels
            int height = 300;      // Fixed height

            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

            for (int i = 0; i < L; i++) {
                // Normal grayscale intensity: start from black (0) and go to white (255)
                int grayValue = i;

                // Scale down if L > 256
                if (L > 256) {
                    grayValue = (int) ((i * 255.0) / (L - 1));
                }

                // Create grayscale color
                Color gray = new Color(grayValue, grayValue, grayValue);

                // Fill the entire column with this color
                for (int y = 0; y < height; y++) {
                    image.setRGB(i, y, gray.getRGB());
                }
            }

            // Save the image to file
            File output = new File("dark_to_light_experiment.png");
            ImageIO.write(image, "png", output);

            System.out.println("✅ Dark-to-light gradient image created successfully!");
        } catch (IOException e) {
        }
    }
}
