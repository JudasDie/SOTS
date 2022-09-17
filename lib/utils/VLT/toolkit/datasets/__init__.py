# from .vot import VOTDataset, VOTLTDataset
from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .nfs import NFSDataset
from .trackingnet import TrackingNetDataset
from .tnl2k import TNL2KDataset
from .got10k import GOT10kDataset
from .nass import NASSDataset
from .nasss import NASSSDataset
from .totb import TOTBDataset
from .otbnl import OTBNLDataset
from .ITB import ITBDataset
from .nassnlp import NASSNLPDataset
from .nllasot import NLLaSOTDataset
from .got10kval import GOT10kvalDataset
from .lasotext import LaSOTEXTDataset

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB100' == name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'LaSOTEXT' == name:
            dataset = LaSOTEXTDataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'NFS' in name:
            dataset = NFSDataset(**kwargs)
        elif 'TrackingNet' == name:
            dataset = TrackingNetDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'GOT-10kval' == name:
            dataset = GOT10kvalDataset(**kwargs)
        elif 'TNL2K' == name:
            dataset = TNL2KDataset(**kwargs)
        elif 'NASS' == name:
            dataset = NASSDataset(**kwargs)
        elif 'NASSS' == name:
            dataset = NASSSDataset(**kwargs)
        elif 'NASSNLP' == name:
            dataset = NASSNLPDataset(**kwargs)
        elif 'OTBNL' == name:
            dataset = OTBNLDataset(**kwargs)
        elif 'ITB' == name:
            dataset = ITBDataset(**kwargs)
        elif 'NLLaSOT' == name:
            dataset = NLLaSOTDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

