# -------------------------------
# Histogram Equalization in R
# -------------------------------

# Install required package if not already installed
if(!require(imager)) install.packages("imager", dependencies=TRUE)
library(imager)

# Step 1: Load the image
img <- load.image("gray.jpg")  # <-- Replace with your image path
plot(img, main="Original Image")

# Step 2: Convert to Grayscale
gray_img <- grayscale(img)
plot(gray_img, main="Grayscale Image")

# Step 3: Convert grayscale image to numeric matrix
img_matrix <- as.data.frame(gray_img)$value

# Step 4: Perform Histogram Equalization manually
# Get histogram information
hist_info <- hist(img_matrix, breaks=256, plot=FALSE)
cdf <- cumsum(hist_info$counts) / sum(hist_info$counts)

# Normalize pixel values
equalized_values <- approx(hist_info$mids, cdf, xout=img_matrix)$y

# Step 5: Convert equalized values back to image
equalized_img <- as.cimg(matrix(equalized_values, nrow=dim(gray_img)[1], ncol=dim(gray_img)[2]))
plot(equalized_img, main="Equalized Image")

# Step 6: Compare Histograms
par(mfrow=c(1,2))
hist(img_matrix, breaks=256, col="gray", main="Original Histogram", xlab="Intensity")
hist(equalized_values, breaks=256, col="gray", main="Equalized Histogram", xlab="Intensity")

