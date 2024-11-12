class StateObserver:
    def __init__(self):
        pass
    def save_to_disk(self, image, episode, step):
        image.save_to_disk(f"A_to_B_GPU_34/images/{episode}/{int(step)}.jpeg")