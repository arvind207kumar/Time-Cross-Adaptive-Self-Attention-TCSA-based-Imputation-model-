import torch
import torch.nn as nn
import torch.nn.functional as F
from Modeling.layers import *


class DualBranchModel(nn.Module):
    def __init__(
        self,
        n_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.input_with_mask = kwargs.get("input_with_mask", True)
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.device = kwargs.get("device", "cuda:1")
        self.n_layers = n_layers

        # Position encoding
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)

        # Temporal branch (upper branch)
        self.temporal_encoder = nn.ModuleList([
            EncoderLayer(
                d_time,
                actual_d_feature,
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                dropout,
                0,
                **kwargs
            ) for _ in range(n_layers)
        ])
        self.X_gc_T = nn.Linear(d_model, d_feature)  # Linear after MMSA

        # Variable branch (lower branch)
        self.variable_encoder = nn.ModuleList([
            EncoderLayer(
                d_feature,
                actual_d_feature,
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                dropout,
                0,
                **kwargs
            ) for _ in range(n_layers)
        ])
        self.X_gc_V = nn.Linear(d_model, d_feature)  # Linear after MMSA

        ## concatination of delta , 

        # Weighted combination layers (C1, C2, C3)
        self.weight_combine_c1 = nn.Sequential(
            nn.Linear(d_feature + d_time, d_feature),
            nn.Sigmoid()
        )
        self.weight_combine_c2 = nn.Sequential(
            nn.Linear(d_feature + d_time, d_feature),
            nn.Sigmoid()
        )
        self.weight_combine_c3 = nn.Sequential(
            nn.Linear(d_feature + d_time, d_feature),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        X, M  = inputs["X"], inputs["missing_mask"]
        delta = inputs['delta_mask']
        I = inputs["indicating_mask"]
        X_hat = inputs["X_hat"]
        
        X_concatenate = torch.cat([delta,M,I], dim=2) 

        # Generate artificial mask indicator I
        #I = torch.bernoulli(0.2 * torch.ones_like(M)).to(M.device)  # 20% random masking
        X_corrupted = M * X + (1 - I) * torch.zeros_like(X)  # Apply corruption based on I
        M = M * (1 - I)  # Update M to account for artificially masked elements
        X_input = torch.cat([X_hat, M], dim=2) if self.input_with_mask else X_corrupted
        # Temporal branch (upper)
        temp_enc_output = self.dropout(self.position_enc(X_input))
        for layer in self.temporal_encoder:
            temp_enc_output, _ = layer(temp_enc_output)
        X_ac_T = self.temporal_gc_linear(temp_enc_output)  # Global correlation

        # Variable branch (lower)
        var_enc_output = self.dropout(self.position_enc(X_input.transpose(1, 2)))
        for layer in self.variable_encoder:
            var_enc_output, _ = layer(var_enc_output)
        X_ac_V = self.variable_gc_linear(var_enc_output).transpose(1, 2)  # Global correlation

        # Weighted combinations
        # C1: Combine X_gc_T and X_ac_V
        c1_weights = self.weight_combine_c1(X_concatenate)
        X_hat_weight = c1_weights * self.X_gc_T + (1 - c1_weights) * X_ac_V

        # C2: Combine X_gc_V and X_gc_T
        c2_weights = self.weight_combine_c2(X_concatenate)
        X_bar = c2_weights * self.X_gc_V + (1 - c2_weights) * X_ac_T

        # C3: Final combination
        c3_weights = self.weight_combine_c3(torch.cat([M, X_corrupted], dim=-1))
        X_prime = c3_weights * X_hat_weight + (1 - c3_weights) * X_bar

        # Replace observed values
        X_imputed = I * X + (1 - I) * X_prime

        return {
            "imputed_data": X_imputed,
            "X_hat": X_hat_weight,
            "X_bar": X_bar,
            "X_prime": X_prime,
            "reconstruction_components": [self.X_gc_T, self.X_gc_V, X_prime]
        }

    def impute(self, inputs):
        """Imputation interface."""
        output_dict = self.forward(inputs)
        return output_dict["imputed_data"], output_dict["reconstruction_components"]

    def calc_loss(self, inputs):
        """Calculate masked reconstruction loss."""
        X, M = inputs["X"], inputs["missing_mask"]
        output_dict = self.forward(inputs)

        reconstruction_loss = 0
        # Calculate loss for each reconstruction component
        for component in output_dict["reconstruction_components"]:
            reconstruction_loss += masked_mae_cal(component, X, M)
        reconstruction_loss /= len(output_dict["reconstruction_components"])

        return {
            "imputed_data": output_dict["imputed_data"],
            "reconstruction_loss": reconstruction_loss,
            "final_reconstruction_MAE": masked_mae_cal(output_dict["X_prime"], X, M)
        }
