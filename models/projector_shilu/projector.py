from torch import nn,Tensor
import torch
import torch.nn.functional as F
from ...models.ll3da.captioner import select_proposal_feature

class projector(nn.Module):
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm is True:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        return self

    def __init__(self, args, in_channels, out_channels, mlp_depth):
        self.freeze_llm = args.freeze_llm
        self.dtype = torch.float16

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_depth = mlp_depth
        self.hidden_size = [1024 * 2 ** i for i in range(mlp_depth)]

        self.box_prompt_projector = self.set_proj()

        self.transformer = AutoModelForCausalLM.from_pretrained(
            args.vocab,
            torch_dtype=self.dtype
        )
        self.n_embd = self.transformer.config.hidden_size

        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
    
    def set_proj(self):
        modules = [nn.Linear(self.in_channels, self.hidden_size[0])]
        for i in range(1, self.mlp_depth):
            modules.append(nn.LayerNorm(self.hidden_size[i - 1]))
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
        modules.append(nn.LayerNorm(self.hidden_size[-1]))
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.hidden_size[-1], self.out_channels))
        return nn.Sequential(*modules)


    def forward(self, detector_output: dict, inputs: dict, is_eval: bool=False) -> dict:
        box_query = inputs.get('box_query', None)       # batch x nquery x 8 x 3
        box_qmask = inputs.get('box_mask', None)        # batch x nquery

        if box_query is not None:
            box_prompt = select_proposal_feature(
                detector_output['prop_features'][-1], 
                detector_output['box_corners'], 
                prop_sem_mask, 
                box_query
            ) # [batch x nquery x n_embd]

            box_prompt = self.box_prompt_projector(box_prompt)

            if not is_eval:
                # 获取文本嵌入
                think_text_ids = inputs.get('think_text_ids', None)
                embedding_layer = self.transformer.get_input_embeddings()
                text_embedding = embedding_layer(think_text_ids)  # [batch x seq_len x n_embd]

                # 计算对比损失
                loss = self.compute_contrastive_loss(box_prompt, text_embedding)

                # 返回结果
                return {
                    'box_prompt': box_prompt,
                    'text_embedding': text_embedding,
                    'loss': loss
                }

            return {'box_prompt': box_prompt}
        
    def compute_contrastive_loss(self, z3D: Tensor, ztext: Tensor) -> Tensor:
        """
        计算对比损失。

        参数:
        - z3D: 3D特征向量，形状为 [batch x nquery x feature_dim]
        - ztext: 文本嵌入，形状为 [batch x seq_len x feature_dim]

        返回:
        - loss: 对比损失
        """
        # 平均池化以减少维度 (例如对nquery或seq_len进行平均)
        z3D = z3D.mean(dim=1)  # [batch x feature_dim]
        ztext = ztext.mean(dim=1)  # [batch x feature_dim]

        # 归一化特征向量
        z3D = F.normalize(z3D, dim=-1)
        ztext = F.normalize(ztext, dim=-1)

        # 计算对比损失
        loss = self.contrastive_loss_fn(z3D, ztext)
        return loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z3D: Tensor, ztext: Tensor) -> Tensor:
        """
        使用InfoNCE Loss计算对比损失。

        参数:
        - z3D: 3D特征向量，形状为 [batch_size, feature_dim]
        - ztext: 文本嵌入，形状为 [batch_size, feature_dim]

        返回:
        - loss: 对比损失
        """
        # 计算余弦相似度矩阵
        sim_matrix = torch.matmul(z3D, ztext.T)  # [batch_size, batch_size]

        # 应用温度参数
        sim_matrix = sim_matrix / self.temperature

        # 创建正样本对的标签
        labels = torch.arange(sim_matrix.size(0)).to(z3D.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
