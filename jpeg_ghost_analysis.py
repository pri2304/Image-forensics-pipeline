from ela_test import ELA

class GHOST:
    @staticmethod
    def jpeg_ghost(img_path):
        metrics, image = ELA.ela(img_path, quality=50)
        return metrics, image

if __name__ == "__main__":
    input_path = "original.jpg"

    jpeg_ghost_result_metrics, jpeg_ghost_result_image = GHOST.jpeg_ghost(input_path)

    print(jpeg_ghost_result_metrics)

    if jpeg_ghost_result_image:
        jpeg_ghost_result_image.save("jpeg_ghost_result.png")
