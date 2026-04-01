import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class exp6ip {

    // Sharpening kernel (mask)
    private static final int[][] SHARPEN_KERNEL = {
        { 0, -1,  0 },
        {-1,  5, -1 },
        { 0, -1,  0 }
    };

    public static void main(String[] args) {
        try {
            // Load grayscale image
            BufferedImage inputImage = ImageIO.read(new File("ex6.jpg"));

            // Sharpen the image
            BufferedImage outputImage = applySharpening(inputImage);

            // Save the output
            ImageIO.write(outputImage, "jpg", new File("sharpened_output.jpg"));
            System.out.println("✅ Output saved as sharpened_output.jpg");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private static BufferedImage applySharpening(BufferedImage input) {
        int width = input.getWidth();
        int height = input.getHeight();
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int newPixel = 0;

                // Apply kernel
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int pixel = input.getRGB(x + kx, y + ky) & 0xFF; // Get grayscale value
                        newPixel += pixel * SHARPEN_KERNEL[ky + 1][kx + 1];
                    }
                }

                // Clamp to [0,255]
                newPixel = Math.max(0, Math.min(255, newPixel));

                // Set new pixel in output image
                int rgb = (newPixel << 16) | (newPixel << 8) | newPixel;
                output.setRGB(x, y, rgb);
            }
        }

        return output;
    }
}
