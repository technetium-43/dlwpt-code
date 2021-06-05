import torch
import os
import imageio


# Chapter 4.7 Exercises

def process_images(directory: str, filetype: str, height: int, width: int) -> torch.tensor:
    """
    Chapter 4.7 Exercises
    1. Take several pictures of red, blue, and green items with your phone or other digital camera
    (or download some from the internet, if a camera isn’t available).
    a) Load each image, and convert it to a tensor.
    b) For each image tensor, use the .mean() method to get a sense of how bright the image is.
    C) Take the mean of each channel of your images. Can you identify the red, green, and blue items
    from only the channel averages? Select a relatively large file containing
    """
    # Load the image filenames
    filenames = [name for name in os.listdir(directory)
                 if os.path.splitext(name)[-1] == filetype]

    # Create batch of size # [N, C, H, W]
    batch_size = len(filenames)
    img_batch = torch.zeros([batch_size, 3, height, width])

    for i, filename in enumerate(filenames):
        # Read the image to a tensor
        img_dir = directory + "/" + filename
        img_np = imageio.imread(img_dir)
        img_np_rgb = img_np[:, :, :3]  # only use RGB channels
        img = torch.from_numpy(img_np_rgb)
        img = img.float()

        # Move the channels so shape = [C, H, W]
        img = img.permute(2, 0, 1)
        print(f"Shape of image [C, H, W]: {img.shape}")

        # Save in the batch of size # [N, C, H, W]
        img_batch[i] = img

    # Normalize the images
    channels = img_batch.shape[1]
    for channel in range(channels):
        channel_mean = img_batch[:, channel, :, :].mean()
        print(f"Channel Mean: {channel_mean}")
        channel_std = img_batch[:, channel, :, :].std()
        img_batch[:, channel] = (img_batch[:, channel] - channel_mean) / channel_std

    return img_batch
