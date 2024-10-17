import os

import matplotlib.pyplot as plt
import numpy as np


def plot_images(
    true_images,
    spn_completions,
    nn_completions,
    dataset_name,
    folder_location,
    evid_percentage,
    plot_middle_line=True,
    horizontal_line=True,
):
    """
    The function `plot_images` takes in true images, SPN completions, and NN completions, and plots them
    side by side with an optional middle line, saving the resulting images in a specified folder.

    :param true_images: The true_images parameter is a numpy array containing the true images. Each
    image should be flattened into a 1D array
    :param spn_completions: The SPN completions are the completed images generated by the SPN model
    :param nn_completions: `nn_completions` is a numpy array containing the completions generated by a
    neural network model. It has shape (num_images, image_size^2), where `num_images` is the number of
    images and `image_size` is the size of each image (assuming square images)
    :param dataset_name: The name of the dataset you are working with. This will be used as part of the
    image file name
    :param folder_location: The folder location where the images will be saved. This should be a string
    representing the path to the folder on your computer where you want to save the images. For example,
    "C:/Users/username/Documents/images"
    :param evid_percentage: The `evid_percentage` parameter represents the percentage of evidence in the
    image. It is used to determine the position of the middle line in the image. For example, if
    `evid_percentage` is set to 0.5, the middle line will be placed at the center of the image
    :param plot_middle_line: The parameter `plot_middle_line` is a boolean value that determines whether
    or not to plot a middle line on the images. If set to `True`, a middle line will be plotted on the
    images. If set to `False`, no middle line will be plotted, defaults to True (optional)
    :param horizontal_line: The `horizontal_line` parameter is a boolean value that determines whether
    the middle line should be a horizontal line (`True`) or a vertical line (`False`). By default, it is
    set to `True`, meaning the middle line will be a horizontal line, defaults to True (optional)
    """
    num_images = spn_completions.shape[0]

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)

    # Reshape the SPN and NN completions into 2D images (assuming square images)
    image_size = int(np.sqrt(spn_completions.shape[1]))
    spn_images = spn_completions.reshape(num_images, image_size, image_size)
    nn_images = nn_completions.reshape(num_images, image_size, image_size)

    # Reshape the true images into 2D images (assuming square images)
    true_images = true_images.reshape(num_images, image_size, image_size)

    for idx in range(num_images):
        spn_image = spn_images[idx]
        nn_image = nn_images[idx]
        true_image = true_images[idx]

        # Plot the images and text below
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Add the middle line
        middle_line = int(image_size * evid_percentage)
        true_image_with_line = np.copy(true_image)
        spn_image_with_line = np.copy(spn_image)
        nn_image_with_line = np.copy(nn_image)
        if plot_middle_line:
            if horizontal_line:
                true_image_with_line[
                    middle_line, :
                ] = 0.5  # Set the middle line to gray
                spn_image_with_line[middle_line, :] = 0.5  # Set the middle line to gray
                nn_image_with_line[middle_line, :] = 0.5  # Set the middle line to gray
            else:
                true_image_with_line[
                    :, middle_line
                ] = 0.5  # Set the middle line to gray
                spn_image_with_line[:, middle_line] = 0.5  # Set the middle line to gray
                nn_image_with_line[:, middle_line] = 0.5  # Set the middle line to gray

        ax[0].imshow(true_image_with_line, cmap="gray")
        ax[1].imshow(spn_image_with_line, cmap="gray")
        ax[2].imshow(nn_image_with_line, cmap="gray")

        # Add text just below the images
        ax[0].text(
            image_size // 2,
            image_size + 3,
            "True",
            fontsize=12,
            color="black",
            ha="center",
        )
        ax[1].text(
            image_size // 2,
            image_size + 3,
            "SPN Completion",
            fontsize=12,
            color="black",
            ha="center",
        )
        ax[2].text(
            image_size // 2,
            image_size + 3,
            "NN Completion",
            fontsize=12,
            color="black",
            ha="center",
        )

        # Hide ticks and labels
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")

        # Save the image inside the specified folder with the specified name
        image_name = f"{dataset_name}_{idx}.png"
        image_path = os.path.join(folder_location, image_name)
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()
