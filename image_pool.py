import random
import torch

class ImagePool:
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than just the ones produced by the latest generators.
    """
    def __init__(self, size):
        self._size = size
        if size > 0:
            self._images = []

    def query(self, imgs):
        """
        Input: a batch of images from the generator.
        Output: a batch of images from the pool or the current batch.
        """
        if self._size <= 0:
            return imgs
        ret_imgs = []
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1] # Preserves (1, C, H, W)
            if len(self._images) < self._size:
                self._images.append(img)
                ret_imgs.append(img)
            else:
                p = random.random()
                if p < 0.5:
                    idx = random.randrange(self._size)
                    ret_imgs.append(self._images[idx])
                    self._images[idx] = img
                else:
                    ret_imgs.append(img)
        return torch.cat(ret_imgs, dim=0)

if __name__ == "__main__":
    pool = ImagePool(5)
    dummy_input = torch.ones((1, 1, 1, 1))
    for i in range(10):
        print(f"Batch {i+1}: Querying with {i}")
        out = pool.query(dummy_input * i)
        print(f"Result: {out.item()}")
