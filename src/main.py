import utils.dataloader as dataloader
import utils.modeller as modeller
import datetime
import utils.visualizer as visualizer

data = dataloader.get("sh000001", datetime.date(2021, 1, 1), datetime.date(2024, 11, 24))
print(data)
visualizer.drawPrior(data)
