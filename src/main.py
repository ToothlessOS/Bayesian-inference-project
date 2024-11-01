import utils.dataloader as dataloader
import utils.visualizer as visualizer

data = dataloader.DataLoader("000001", "20200101", "20241031").get()
visualizer.draw(data)