import torch
from torch import optim, nn, inference_mode, softmax, argmax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
def train_step(
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss = 0
    for batch, (X, magnitude, useless_label) in enumerate(dataloader):
        X = X.to(device)
        X_reconstructed = model(X)
        loss = loss_fn(X_reconstructed, X)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(dataloader)
    return train_loss

def test_step(
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    with inference_mode():
        for batch, (X, magnitude, useless_label)  in enumerate(dataloader):
            X = X.to(device)
            X_reconstructed = model(X)
            loss = loss_fn(X_reconstructed, X)
            test_loss += loss
    test_loss = test_loss/len(dataloader)
    return test_loss
    
def train(
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epochs: int
) -> Dict[str, List]:
    
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': [],
               }
    writer = SummaryWriter()
    model.to(device)
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )
        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        print(f"Epoch: {epoch+1}")
        print(f"Train loss: {train_loss}.")
        print(f"Test loss: {test_loss}.")
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(16, 1, 30, 25).to(device))
    
    # Close the writer
    writer.close()
    return results

    



