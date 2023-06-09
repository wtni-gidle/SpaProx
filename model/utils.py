import torch



def to_device(
    data, 
    device: torch.device
):
    """
    Move tensor(s) to chosen `device`.

    Parameters
    ----------
    data : Any
        The tensor(s) to move.

    device : torch.device
        The desired `device`.

    Returns
    -------
    Return the moved data.
    """

    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking = True)



class DeviceDataLoader():
    """
    DeviceDataLoader
    """

    def __init__(
        self, 
        dl, 
        device
    ):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dl)



class Accumulator:
    """
    Accumulator
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, id):
        return self.data[id]


