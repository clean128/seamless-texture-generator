import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Ensure TensorFlow uses GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class SeamlessTextureGenerator:
    def __init__(self, input_size=256, latent_dim=128):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.build_model()
        
    def build_encoder(self):
        encoder_input = Input(shape=(self.input_size, self.input_size, 3), name='encoder_input')
        
        # Convolutional layers
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        # Flatten and encode to latent space
        x = layers.Flatten()(x)
        encoder_output = layers.Dense(self.latent_dim)(x)
        
        return Model(encoder_input, encoder_output, name='encoder')
    
    def build_decoder(self):
        latent_input = Input(shape=(self.latent_dim,), name='latent_input')
        
        # Reshape to start the convolutional transpose process
        x = layers.Dense(16 * 16 * 512)(latent_input)
        x = layers.Reshape((16, 16, 512))(x)
        
        # Transposed convolutions
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        # Final layer with sigmoid activation to output image
        decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', strides=2)(x)
        
        return Model(latent_input, decoder_output, name='decoder')
    
    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Connect encoder and decoder to create autoencoder
        autoencoder_input = Input(shape=(self.input_size, self.input_size, 3), name='autoencoder_input')
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(autoencoder_input, decoded, name='autoencoder')
        
        # Custom loss function for seamless texture generation
        self.autoencoder.compile(
            optimizer='adam',
            loss=self.seamless_loss
        )
        
        print("Model built successfully")
    
    def seamless_loss(self, y_true, y_pred):
        """
        Custom loss function that combines:
        1. Reconstruction loss (MSE)
        2. Edge continuity loss (makes opposite edges similar)
        3. Texture pattern preservation loss
        """
        # Standard reconstruction loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Edge continuity loss
        # Make left edge similar to right edge, and top edge similar to bottom edge
        left_edge = y_pred[:, :, 0, :]
        right_edge = y_pred[:, :, -1, :]
        top_edge = y_pred[:, 0, :, :]
        bottom_edge = y_pred[:, -1, :, :]
        
        edge_loss = tf.reduce_mean(tf.square(left_edge - right_edge)) + \
                    tf.reduce_mean(tf.square(top_edge - bottom_edge))
        
        # Structure preservation loss using gradient information
        # This helps maintain the texture patterns while making it seamless
        y_true_dx, y_true_dy = tf.image.image_gradients(y_true)
        y_pred_dx, y_pred_dy = tf.image.image_gradients(y_pred)
        
        gradient_loss = tf.reduce_mean(tf.square(y_true_dx - y_pred_dx)) + \
                        tf.reduce_mean(tf.square(y_true_dy - y_pred_dy))
        
        # Weight the different loss components
        total_loss = mse_loss + 0.5 * edge_loss + 0.3 * gradient_loss
        
        return total_loss
    
    def prepare_data(self, image_dir, augment=True, validation_split=0.2):
        """Prepare training data from a directory of texture images"""
        images = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Load all images from the directory
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                img_path = os.path.join(image_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.input_size, self.input_size), Image.LANCZOS)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    
                    # Data augmentation
                    if augment:
                        # Flip horizontally
                        images.append(np.fliplr(img_array))
                        # Flip vertically
                        images.append(np.flipud(img_array))
                        # Rotate 90 degrees
                        images.append(np.rot90(img_array))
                        # Rotate 180 degrees
                        images.append(np.rot90(img_array, 2))
                        # Rotate 270 degrees
                        images.append(np.rot90(img_array, 3))
                        
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError(f"No valid images found in {image_dir}")
        
        # Convert to numpy array
        images = np.array(images)
        
        # Split into training and validation sets
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        
        # Ensure at least one validation image
        val_size = max(1, int(validation_split * len(images)))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_images = images[train_indices]
        val_images = images[val_indices]
        
        print(f"Prepared {len(train_images)} training images and {len(val_images)} validation images")
        
        return train_images, val_images
    
    def train(self, train_data, validation_data=None, epochs=100, batch_size=16):
        """Train the model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        if validation_data is not None and len(validation_data) > 0:
            val_data = (validation_data, validation_data)
        else:
            val_data = None
        
        history = self.autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=val_data,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, model_dir):
        """Save the encoder and decoder models"""
        os.makedirs(model_dir, exist_ok=True)
        self.encoder.save(os.path.join(model_dir, 'encoder.h5'))
        self.decoder.save(os.path.join(model_dir, 'decoder.h5'))
        print(f"Models saved to {model_dir}")
    
    def load_model(self, model_dir):
        """Load the encoder and decoder models"""
        self.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.h5'))
        self.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'decoder.h5'))
        
        # Connect encoder and decoder to create autoencoder
        autoencoder_input = Input(shape=(self.input_size, self.input_size, 3), name='autoencoder_input')
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(autoencoder_input, decoded, name='autoencoder')
        
        # Compile with the custom loss function
        self.autoencoder.compile(
            optimizer='adam',
            loss=self.seamless_loss
        )
        
        print(f"Models loaded from {model_dir}")
    
    def generate_seamless_texture(self, input_image_path, output_image_path=None, blend_edges=True, post_process=True):
        """Generate a seamless texture from an input image"""
        # Load and prepare input image
        input_img = Image.open(input_image_path).convert('RGB')
        input_img = input_img.resize((self.input_size, self.input_size), Image.LANCZOS)
        input_array = np.array(input_img) / 255.0
        input_batch = np.expand_dims(input_array, axis=0)
        
        # Generate seamless texture using the autoencoder
        seamless_array = self.autoencoder.predict(input_batch)[0]
        
        # Optional post-processing to further improve seamless quality
        if post_process:
            seamless_array = self.post_process_texture(seamless_array)
        
        # Additional edge blending if requested
        if blend_edges:
            seamless_array = self.blend_edges(seamless_array)
        
        # Convert to image
        seamless_img = Image.fromarray((seamless_array * 255).astype(np.uint8))
        
        # Save if output path is provided
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
            seamless_img.save(output_image_path)
            print(f"Seamless texture saved to {output_image_path}")
        
        return seamless_img
    
    def post_process_texture(self, texture_array):
        """
        Apply post-processing to further improve seamless quality:
        1. Frequency domain filtering to remove seam artifacts
        2. Color harmonization
        3. Detail preservation
        """
        # Convert to OpenCV format (0-255, BGR)
        img = (texture_array * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Split into channels
        channels = cv2.split(img_bgr)
        processed_channels = []
        
        for channel in channels:
            # Apply FFT
            dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Create a mask (high pass filter with notch filter for periodic artifacts)
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create a high pass filter
            mask = np.ones((rows, cols, 2), np.float32)
            r = 10  # Low frequency cutoff radius
            center_mask = np.zeros((rows, cols), np.float32)
            cv2.circle(center_mask, (ccol, crow), r, 1, -1)
            mask[:,:,0] = mask[:,:,0] * (1 - center_mask)
            mask[:,:,1] = mask[:,:,1] * (1 - center_mask)
            
            # Apply notch filters at frequencies that often cause visible seams
            for i in range(1, 4):
                for j in range(1, 4):
                    notch_r = 5
                    cv2.circle(mask, (ccol + cols//i, crow), notch_r, 0, -1)
                    cv2.circle(mask, (ccol - cols//i, crow), notch_r, 0, -1)
                    cv2.circle(mask, (ccol, crow + rows//j), notch_r, 0, -1)
                    cv2.circle(mask, (ccol, crow - rows//j), notch_r, 0, -1)
            
            # Apply mask and inverse DFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            
            # Normalize the image
            min_val = np.min(img_back)
            max_val = np.max(img_back)
            if max_val > min_val:
                img_back = (img_back - min_val) / (max_val - min_val) * 255
            
            processed_channels.append(img_back.astype(np.uint8))
        
        # Merge channels
        processed_img = cv2.merge(processed_channels)
        
        # Apply detail-preserving filter
        processed_img = cv2.detailEnhance(processed_img, sigma_s=10, sigma_r=0.15)
        
        # Convert back to RGB and normalize to 0-1
        result = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB) / 255.0
        
        return result
    
    def blend_edges(self, texture_array):
        """Apply additional edge blending to ensure perfect tiling"""
        h, w, c = texture_array.shape
        result = texture_array.copy()
        
        # Blend width (pixels from each edge)
        blend_width = int(w * 0.05)  # 5% of the width
        
        # Create weight matrices for blending
        x_weights = np.ones((h, w))
        y_weights = np.ones((h, w))
        
        # X-direction weights (left-right blending)
        for i in range(blend_width):
            weight = i / blend_width
            x_weights[:, i] = weight
            x_weights[:, w-i-1] = weight
        
        # Y-direction weights (top-bottom blending)
        for i in range(blend_width):
            weight = i / blend_width
            y_weights[i, :] = weight
            y_weights[h-i-1, :] = weight
        
        # Combined weights
        combined_weights = np.minimum(x_weights, y_weights)
        combined_weights = np.expand_dims(combined_weights, axis=2)
        combined_weights = np.repeat(combined_weights, c, axis=2)
        
        # Create a wrapped version of the texture for blending
        wrapped = np.zeros_like(texture_array)
        
        # Copy the opposite edges
        wrapped[:blend_width, :, :] = texture_array[h-blend_width:, :, :]  # Top edge gets bottom content
        wrapped[h-blend_width:, :, :] = texture_array[:blend_width, :, :]  # Bottom edge gets top content
        wrapped[:, :blend_width, :] = texture_array[:, w-blend_width:, :]  # Left edge gets right content
        wrapped[:, w-blend_width:, :] = texture_array[:, :blend_width, :]  # Right edge gets left content
        
        # Handle the corners specially
        wrapped[:blend_width, :blend_width, :] = texture_array[h-blend_width:, w-blend_width:, :]  # Top-left
        wrapped[:blend_width, w-blend_width:, :] = texture_array[h-blend_width:, :blend_width, :]  # Top-right
        wrapped[h-blend_width:, :blend_width, :] = texture_array[:blend_width, w-blend_width:, :]  # Bottom-left
        wrapped[h-blend_width:, w-blend_width:, :] = texture_array[:blend_width, :blend_width, :]  # Bottom-right
        
        # Blend using weights
        result = result * combined_weights + wrapped * (1 - combined_weights)
        
        return result
    
    def evaluate_seamlessness(self, texture_array):
        """
        Evaluate how seamless a texture is by comparing edges
        Returns a score between 0-1 where 1 is perfectly seamless
        """
        h, w, c = texture_array.shape
        
        # Get edges
        left_edge = texture_array[:, 0, :]
        right_edge = texture_array[:, -1, :]
        top_edge = texture_array[0, :, :]
        bottom_edge = texture_array[-1, :, :]
        
        # Calculate SSIM for opposite edges
        h_score, _ = ssim(left_edge, right_edge, multichannel=True, full=True)
        v_score, _ = ssim(top_edge, bottom_edge, multichannel=True, full=True)
        
        # Calculate corner matching scores
        corners = [
            (texture_array[0, 0, :], texture_array[0, -1, :], texture_array[-1, 0, :], texture_array[-1, -1, :])
        ]
        
        corner_diffs = [
            np.mean(np.abs(corners[0][0] - corners[0][3])),
            np.mean(np.abs(corners[0][1] - corners[0][2]))
        ]
        corner_score = 1 - np.mean(corner_diffs)
        
        # Combine scores (weighted average)
        combined_score = 0.4 * h_score + 0.4 * v_score + 0.2 * corner_score
        
        return {
            'horizontal_seamlessness': float(h_score),
            'vertical_seamlessness': float(v_score),
            'corner_seamlessness': float(corner_score),
            'overall_seamlessness': float(combined_score)
        }
    
    def batch_process(self, input_dir, output_dir):
        """Process all image files in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"seamless_{filename}")
                
                try:
                    print(f"Processing {filename}...")
                    self.generate_seamless_texture(input_path, output_path)
                    count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print(f"Successfully processed {count} texture images")

# Example usage
def main():
    # Create generator
    generator = SeamlessTextureGenerator(input_size=256)
    
    # Check if model exists, otherwise train
    model_dir = "models/seamless_texture"
    
    if os.path.exists(os.path.join(model_dir, 'encoder.h5')):
        print("Loading existing model...")
        generator.load_model(model_dir)
    else:
        print("Training new model...")
        # Prepare training data
        train_data, val_data = generator.prepare_data("input_textures")
        
        # Train the model
        history = generator.train(train_data, val_data, epochs=100, batch_size=16)
        
        # Save the model
        generator.save_model(model_dir)
    
    # Process a single texture
    generator.generate_seamless_texture(
        "input_textures/texture_dataset.jpg", 
        "output_textures/seamless_texture_dataset.jpg"
    )
    
    # Or batch process a directory
    # generator.batch_process("input_textures", "output_textures")

if __name__ == "__main__":
    main()