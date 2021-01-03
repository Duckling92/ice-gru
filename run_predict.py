from modules.model import IceGRU
from modules.utilities import get_project_root
from pathlib import Path


def load_events():
    pass


model_path = Path(get_project_root(), "models/direction")
model = IceGRU(model_path)
events = load_events()
predictions = model.predict(events)
