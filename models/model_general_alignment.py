import torch
import importlib
from torch import nn

class ProjectNet(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector is True:
            self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False
        return self
    
    def pretrained_parameters(self):
        if hasattr(self.projector, 'pretrained_parameters'):
            return self.projector.pretrained_parameters()
        else:
            return []
    
    def __init__(self, args, dataset_config, train_dataset):
        super(ProjectNet, self).__init__()
        
        self.freeze_detector = args.freeze_detector
        self.detector = None
        self.projector = None
        
        if args.detector is not None:
            detector_module = importlib.import_module(
                f'models.{args.detector}.detector'
            )
            self.detector = detector_module.detector(args, dataset_config)
        
        if args.projector is not None:
            projector = importlib.import_module(
                f'models.{args.projector}.projector'
            )
            self.projector = projector_module.projector(args, train_dataset)
        
        self.train()
        
    def forward(self, batch_data_label: dict, is_eval: bool=False) -> dict:
        
        outputs = {'loss': torch.zeros(1)[0].cuda()}
        
        if self.detector is not None:
            if self.freeze_detector is True:
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                outputs = self.detector(batch_data_label, is_eval=is_eval)
                
        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()
        
        if self.projector is not None:
            outputs = self.projector(
                outputs, 
                batch_data_label, 
                is_eval=is_eval, 
            )
        else:
            batch, nproposals, _, _ = outputs['box_corners'].shape
            outputs['lang_cap'] = [
                ["this is a valid match!"] * nproposals
            ] * batch
        return outputs
