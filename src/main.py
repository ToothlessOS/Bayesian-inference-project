import utils.dataloader as dataloader
import datetime
import utils.visualizer as visualizer

data = dataloader.get("sh000001", datetime.date(2021, 1, 1), datetime.date(2024, 11, 9))
print(data)
visualizer.drawPrior(data)