import torch
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5):
        """
        Combina duas funções de perda com pesos personalizados.

        Args:
            loss1 (nn.Module): Primeira função de perda (ex: FocalLoss).
            loss2 (nn.Module): Segunda função de perda (ex: DiceLoss).
            weight1 (float): Peso para a primeira função de perda.
            weight2 (float): Peso para a segunda função de perda.
        """
        super(JointLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, outputs, targets):
        # Calcula as duas perdas
        loss1 = self.loss1(outputs, targets)
        loss2 = self.loss2(outputs, targets)
        
        # Combina as perdas com os pesos
        total_loss = self.weight1 * loss1 + self.weight2 * loss2
        return total_loss