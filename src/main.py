import utils.dataloader as dataloader
import utils.visualizer as visualizer

data = dataloader.DataLoader("601318", "20231191", "20241031").get()
visualizer.draw(data)