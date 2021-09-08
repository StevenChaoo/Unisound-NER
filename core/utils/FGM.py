import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        #print(self.model.named_parameters())
        #flagg = False
        for name, param in self.model.named_parameters():
            #print(name)
            if param.requires_grad and emb_name in name:
                #flagg = True
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
        #print(flagg)

    def restore(self, emb_name= 'bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
