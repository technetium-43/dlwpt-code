from p1ch4.process_images import process_images


def test_process_images():
    print("Running process_images() for Exercise 1")
    img_batch = process_images(directory="../../data/p1ch4/my-images/", filetype=".png", height=750, width=1_000)
    print(f"Image batch of [N, C, H, W]: {img_batch.size()}")


