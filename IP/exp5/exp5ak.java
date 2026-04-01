import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class exp5ak {
    public static void main(String[] args) {
        String inputImagePath = "input4.png";          // Input image filename
        String outputHistogramPath = "histogram_with_axes.png"; // Output filename

        try {
            BufferedImage inputImage = ImageIO.read(new File(inputImagePath));
            if (inputImage == null) {
                System.err.println("❌ Error: Could not read image file " + inputImagePath);
                return;
            }

            BufferedImage histImage = createHistogramImage(inputImage);
            ImageIO.write(histImage, "png", new File(outputHistogramPath));
            System.out.println("✅ Histogram with axes saved as " + outputHistogramPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static BufferedImage createHistogramImage(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Step 1: Compute grayscale histogram
        int[] histogram = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);
                histogram[gray]++;
            }
        }

        // Step 2: Find max value
        int max = 0;
        for (int val : histogram) {
            if (val > max) max = val;
        }

        // Step 3: Create histogram image with margins for axes
        int histWidth = 300;   // 256 + margins
        int histHeight = 250;  // 200 + margins
        int marginLeft = 40;
        int marginBottom = 30;

        BufferedImage histImage = new BufferedImage(histWidth, histHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = histImage.createGraphics();

        // White background
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, histWidth, histHeight);

        // Draw axes
        g.setColor(Color.BLACK);
        int xAxisY = histHeight - marginBottom;
        int yAxisX = marginLeft;

        g.drawLine(yAxisX, 10, yAxisX, xAxisY);                 // Y-axis
        g.drawLine(yAxisX, xAxisY, histWidth - 10, xAxisY);     // X-axis

        // Step 4: Draw histogram bars
        g.setColor(Color.DARK_GRAY);
        for (int i = 0; i < 256; i++) {
            int barHeight = (int)(((double)histogram[i] / max) * (histHeight - marginBottom - 20));
            int x = yAxisX + i;
            int y = xAxisY - barHeight;
            g.drawLine(x, xAxisY, x, y);
        }

        // Step 5: Draw X-axis ticks (intensity levels)
        g.setColor(Color.BLACK);
        g.setFont(new Font("Arial", Font.PLAIN, 10));
        for (int i = 0; i <= 255; i += 50) {
            int x = yAxisX + i;
            g.drawLine(x, xAxisY, x, xAxisY + 5);
            g.drawString(String.valueOf(i), x - 5, xAxisY + 20);
        }

        // Step 6: Draw Y-axis ticks (frequency scale)
        for (int j = 0; j <= 5; j++) {
            int value = max * j / 5;
            int y = xAxisY - (int)((double)(j) / 5 * (histHeight - marginBottom - 20));
            g.drawLine(yAxisX - 5, y, yAxisX, y);
            g.drawString(String.valueOf(value), 5, y + 5);
        }

        // Step 7: Add axis labels
        g.setFont(new Font("Arial", Font.BOLD, 12));
        g.drawString("Pixel Intensity (0–255)", histWidth / 2 - 50, histHeight - 5);

        // Rotate text for Y-axis label
        g.rotate(-Math.PI / 2);
        g.drawString("Frequency", -histHeight / 2 - 30, 15);
        g.rotate(Math.PI / 2);  // Restore rotation

        g.dispose();
        return histImage;
    }
}
