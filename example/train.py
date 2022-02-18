from mohou.types import MultiEpisodeDataChunk
from mohou.dataset import AutoEncoderDataset
from mohou.file import load_object

chunk = load_object(MultiEpisodeDataChunk, 'kuka_reaching')
dataset = AutoEncoderDataset.from_chunk(chunk)
print(dataset[0])
