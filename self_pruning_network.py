"""
The Self-Pruning Neural Network (Smooth Gradual Sparsity Edition)
Author: Candidate
Tredence Analytics AI Engineering Intern Assignment

This script implements a self-pruning neural network with controlled, stable pruning behavior.
It utilizes a Delayed Sparsity (Warmup) schedule and balanced learning rates to ensure the model
gradually discovers and prunes unneeded weights, yielding a beautiful sparsity-accuracy tradeoff curve.
"""

import math
import json
import logging
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights dynamically during training.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate scores to 1.0 (~0.73 sigmoid value)
        nn.init.constant_(self.gate_scores, 1.0)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.get_gates()
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)


class SelfPruningNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                l1_loss += torch.sum(module.get_gates())
        return l1_loss

    def get_sparsity_metrics(self, threshold: float = 1e-2) -> Tuple[float, np.ndarray]:
        total_weights = 0
        pruned_weights = 0
        all_gates = []
        
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates().detach()
                total_weights += gates.numel()
                pruned_weights += torch.sum(gates < threshold).item()
                all_gates.append(gates.cpu().numpy().flatten())
                
        sparsity_pct = (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0
        return sparsity_pct, np.concatenate(all_gates)


def train_model(target_lmbda: float, epochs: int = 12, warmup_epochs: int = 3, device: torch.device = torch.device("cpu"), batch_size: int = 256) -> Tuple[float, float, np.ndarray]:
    logging.info(f"Starting training with target λ = {target_lmbda}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SelfPruningNet().to(device)
    classification_criterion = nn.CrossEntropyLoss()
    
    # CONTROLLED OPTIMIZER IMBALANCE:
    # 0.05 was too aggressive, causing immediate 99% collapse. 
    # We lower the gate_lr to 0.015 so sparsity descends smoothly and only captures the truly weak connections!
    gate_params = [p for n, p in model.named_parameters() if 'gate_score' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_score' not in n]
    
    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 2e-3},
        {'params': gate_params, 'lr': 1.5e-2}
    ])

    for epoch in range(epochs):
        model.train()
        running_loss, running_cls_loss, running_sparse_loss = 0.0, 0.0, 0.0
        
        # DELAYED SPARSITY WARMUP:
        # Gradually increase Lambda from 0.0 so the network learns the important connections FIRST before pruning them.
        current_lmbda = target_lmbda * min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else target_lmbda
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            cls_loss = classification_criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()
            
            loss = cls_loss + current_lmbda * sparsity_loss
            loss.backward()
            
            # Application of gradient clipping to prevent erratic massive gate drops
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_sparse_loss += sparsity_loss.item()
            
        logging.info(
            f"Epoch {epoch+1:02d}/{epochs} [λ={current_lmbda:.5f}] - "
            f"Total Loss: {running_loss/len(trainloader):.4f} | "
            f"CE: {running_cls_loss/len(trainloader):.4f} | "
            f"Sparsity L1: {running_sparse_loss/len(trainloader):.1f}"
        )
              
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    sparsity_pct, all_gates = model.get_sparsity_metrics(threshold=1e-2)
    
    logging.info(f"=> Accuracy: {test_accuracy:.2f}% | Sparsity (< 1e-2): {sparsity_pct:.2f}%\\n")
    return test_accuracy, sparsity_pct, all_gates


def visualize_gradual_tradeoff(lambdas: List[float], accuracies: List[float], sparsities: List[float]) -> None:
    """Plots the smooth Sparsity vs. Accuracy tradeoff curve."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Convert lambdas to strings so they space out evenly (categorical) on the x-axis instead of bunching up
    lambda_labels = [f"{lmbda}" for lmbda in lambdas]
    
    ax1.set_xlabel('Lambda Penalty')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue', fontweight='bold')
    ax1.plot(lambda_labels, accuracies, marker='o', color='tab:blue', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sparsity (%)', color='tab:red', fontweight='bold')
    ax2.plot(lambda_labels, sparsities, marker='s', color='tab:red', linewidth=2, linestyle='dashed', label='Sparsity')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title("Gradual Sparsity vs Accuracy Tradeoff", fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig('sparsity_vs_accuracy.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Initialized PyTorch Using Accelerator: {device.__class__.__name__} ({device})")

    # New Gradual Sweep parameters requested by the critique
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4] 
    results = {}
    
    acc_list, sparse_list = [], []
    best_lambda, best_gates = None, None
    
    for lmbda in lambdas:
        acc, spars, gates = train_model(lmbda, epochs=12, warmup_epochs=3, device=device)
        results[lmbda] = {"accuracy": acc, "sparsity": spars}
        acc_list.append(acc)
        sparse_list.append(spars)
        
        # Keep track of a highly pruned but stable representation
        if spars > 50.0:
            best_lambda = lmbda
            best_gates = gates
            
    logging.info("--- Smooth Self-Pruning Tradeoff Results ---")
    for lmbda in lambdas:
        logging.info(f"Lambda: {lmbda:<8} | Accuracy: {results[lmbda]['accuracy']:>5.2f}% | Sparsity (< 1e-2): {results[lmbda]['sparsity']:>5.2f}%")
        
    visualize_gradual_tradeoff(lambdas, acc_list, sparse_list)
    
    if best_gates is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(best_gates, bins=50, alpha=0.75, color='indigo', edgecolor='black', label=f'Best Smooth Gates (λ={best_lambda})')
        plt.title(f"Distribution of Final Gating Values Across Network", fontsize=14, fontweight='bold')
        plt.xlabel("Gate Value (0.0 = completely pruned, 1.0 = fully active)", fontsize=12)
        plt.ylabel("Parameter Frequency", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('sparsity_distribution.png', dpi=300)
        plt.close()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
