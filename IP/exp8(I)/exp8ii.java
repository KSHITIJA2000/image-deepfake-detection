import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class exp8ii {
    @SuppressWarnings({"UseSpecificCatch", "CallToPrintStackTrace"})
    public static void main(String[] args) {
        try {
            // Load input image
            BufferedImage input = ImageIO.read(new File("ak1.png"));
            int width = input.getWidth();
            int height = input.getHeight();

            // Convert to grayscale
            BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = input.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    int grayVal = (r + g + b) / 3;
                    int newRGB = (grayVal << 16) | (grayVal << 8) | grayVal;
                    gray.setRGB(x, y, (0xFF << 24) | newRGB);
                }
            }

            // Perform Opening (Erosion -> Dilation)
            BufferedImage eroded = erode(gray, 3);
            BufferedImage opened = dilate(eroded, 3);
            ImageIO.write(opened, "jpg", new File("output_opening.jpg"));

            // Perform Closing (Dilation -> Erosion)
            BufferedImage dilated = dilate(gray, 3);
            BufferedImage closed = erode(dilated, 3);
            ImageIO.write(closed, "jpg", new File("output_closing.jpg"));

            System.out.println("✅ Opening and Closing operations completed successfully!");
            System.out.println("📁 Results saved as output_opening.jpg and output_closing.jpg");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // --- Morphological Erosion ---
    public static BufferedImage erode(BufferedImage img, int kernelSize) {
        int width = img.getWidth();
        int height = img.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        int offset = kernelSize / 2;

        for (int y = offset; y < height - offset; y++) {
            for (int x = offset; x < width - offset; x++) {
                int min = 255;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int pixel = img.getRGB(x + kx, y + ky) & 0xFF;
                        if (pixel < min) min = pixel;
                    }
                }
                int newRGB = (min << 16) | (min << 8) | min;
                result.setRGB(x, y, (0xFF << 24) | newRGB);
            }
        }
        return result;
    }

    // --- Morphological Dilation ---
    public static BufferedImage dilate(BufferedImage img, int kernelSize) {
        int width = img.getWidth();
        int height = img.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        int offset = kernelSize / 2;

        for (int y = offset; y < height - offset; y++) {
            for (int x = offset; x < width - offset; x++) {
                int max = 0;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int pixel = img.getRGB(x + kx, y + ky) & 0xFF;
                        if (pixel > max) max = pixel;
                    }
                }
                int newRGB = (max << 16) | (max << 8) | max;
                result.setRGB(x, y, (0xFF << 24) | newRGB);
            }
        }
        return result;
    }
}
