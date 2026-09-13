"""Trainable four-layer MaAS policy controller."""
from __future__ import annotations
from typing import Any, Sequence
from ..contracts import EARLY_STOP_OPERATOR, GENERATE_OPERATOR, OPERATOR_SELECTION_THRESHOLD, FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT

def _torch():
    try:
        import torch, torch.nn.functional as F
        return torch, F
    except ImportError as exc: raise ImportError("MultiLayerController requires PyTorch") from exc

class OperatorSelector:
    def __new__(cls, *args, **kwargs):
        torch, _ = _torch()
        class _Selector(torch.nn.Module):
            def __init__(self, input_dim=384, hidden_dim=32, is_first_layer=False):
                super().__init__(); self.is_first_layer=is_first_layer
                self.operator_encoder=torch.nn.Linear(input_dim if is_first_layer else input_dim*2, hidden_dim)
                self.query_encoder=torch.nn.Linear(input_dim, hidden_dim)
            def forward(self, query, operators, previous=None):
                _, F = _torch(); query= query.unsqueeze(0) if query.dim()==1 else query
                q=F.normalize(self.query_encoder(query),p=2,dim=1)
                op=operators
                if previous is not None and not self.is_first_layer:
                    op=torch.cat([op, previous[0].unsqueeze(0).expand(op.size(0),-1)],dim=1)
                op=F.normalize(self.operator_encoder(op),p=2,dim=1); scores=q@op.T
                return torch.log_softmax(scores,dim=1), torch.softmax(scores,dim=1)
        return _Selector(*args, **kwargs)

class MultiLayerController:
    def __new__(cls, input_dim=384, hidden_dim=32, num_layers=4, device=None, embedding_provider=None):
        torch, _ = _torch()
        class _Controller(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.device=device or torch.device("cpu"); self.embedding_provider=embedding_provider
                self.layers=torch.nn.ModuleList([OperatorSelector(input_dim,hidden_dim,is_first_layer=i==0) for i in range(num_layers)])
            def forward(self, query: str, operators_embedding, selection_operator_names: Sequence[str], log_path=None):
                if self.embedding_provider is None: raise ValueError("embedding_provider is required")
                q=self.embedding_provider.encode(query).to(self.device); ops=operators_embedding.to(self.device); logs=[]; names=[]; prev=None
                for i, layer in enumerate(self.layers):
                    lp, probs=layer(q,ops,prev); p=probs.squeeze(0); selected=torch.where(p >= OPERATOR_SELECTION_THRESHOLD)[0]
                    if selected.numel()==0: selected=torch.multinomial(p,1)
                    selected_names=[selection_operator_names[int(x)] for x in selected]
                    penalty=False
                    if i==0 and any(n.lower()==EARLY_STOP_OPERATOR.lower() for n in selected_names):
                        selected=torch.tensor([selection_operator_names.index(GENERATE_OPERATOR)],device=self.device); selected_names=[GENERATE_OPERATOR]; penalty=True
                    if i==0 and GENERATE_OPERATOR not in selected_names:
                        selected=torch.tensor([selection_operator_names.index(GENERATE_OPERATOR)],device=self.device); selected_names=[GENERATE_OPERATOR]
                    value=lp.squeeze(0)[selected].sum()
                    if penalty: value=value + torch.as_tensor(FIRST_LAYER_EARLY_STOP_LOG_PROB_ADJUSTMENT,device=value.device,dtype=value.dtype)
                    logs.append(value); names.append(selected_names); prev=ops[selected]
                    if penalty or any(n.lower()==EARLY_STOP_OPERATOR.lower() for n in selected_names): break
                return logs, names
        return _Controller()

__all__ = ["OperatorSelector", "MultiLayerController"]
