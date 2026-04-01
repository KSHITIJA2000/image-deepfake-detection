import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class exp7ip {

    // --- 3x3 Filter Kernels ---

    // 1. Simple Averaging Filter (Mean Filter)
    private static final int[][] SIMPLE_AVERAGE_KERNEL = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    private static final int SIMPLE_AVERAGE_DIVISOR = 9;

    // 2. Weighted Averaging Filter (Gaussian-like Filter)
    private static final int[][] WEIGHTED_AVERAGE_KERNEL = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    private static final int WEIGHTED_AVERAGE_DIVISOR = 16;

    public static void main(String[] args) {
        String inputImagePath = "ex7.jpg"; // Must exist in same directory

        try {
            // 1. Read the image
            BufferedImage originalImage = ImageIO.read(new File(inputImagePath));
            if (originalImage == null) {
                System.out.println("Error: Image not found or could not be read at: " + inputImagePath);
                return;
            }

            // 2. Apply Simple Averaging Filter (Blur)
            applyFilter(originalImage, SIMPLE_AVERAGE_KERNEL, SIMPLE_AVERAGE_DIVISOR, "simple_filtered.jpg");

            // 3. Apply Weighted Averaging Filter (Gaussian-like)
            applyFilter(originalImage, WEIGHTED_AVERAGE_KERNEL, WEIGHTED_AVERAGE_DIVISOR, "weighted_filtered.jpg");

            System.out.println("✅ Filtering complete.");
            System.out.println("Output saved as: simple_filtered.jpg");
            System.out.println("Output saved as: weighted_filtered.jpg");

        } catch (IOException e) {
            System.err.println("❌ Error: " + e.getMessage());
        }
    }

    /**
     * Applies a 3x3 filter kernel to the image.
     * Works on all color channels (R, G, B).
     */
    public static BufferedImage applyFilter(BufferedImage original, int[][] kernel, int divisor, String outputFileName) throws IOException {
        int width = original.getWidth();
        int height = original.getHeight();

        BufferedImage filteredImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        // Skip border pixels for simplicity
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {

                long sumR = 0, sumG = 0, sumB = 0;

                // Apply kernel on 3x3 neighborhood
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        int rgb = original.getRGB(x + i, y + j);
                        Color color = new Color(rgb);

                        int r = color.getRed();
                        int g = color.getGreen();
                        int b = color.getBlue();

                        int weight = kernel[j + 1][i + 1];

                        sumR += r * weight;
                        sumG += g * weight;
                        sumB += b * weight;
                    }
                }

                int newR = (int)(sumR / divisor);
                int newG = (int)(sumG / divisor);
                int newB = (int)(sumB / divisor);

                // Clamp pixel values
                newR = Math.max(0, Math.min(255, newR));
                newG = Math.max(0, Math.min(255, newG));
                newB = Math.max(0, Math.min(255, newB));

                filteredImage.setRGB(x, y, new Color(newR, newG, newB).getRGB());
            }
        }

        ImageIO.write(filteredImage, "jpg", new File(outputFileName));
        return filteredImage;
    }
}
