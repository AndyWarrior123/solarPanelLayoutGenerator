import torch
from torch.utils.data import DataLoader
from src.dataset import make_datasets
from src.model import SolarUNet
from src.loss import CombinedLoss
from src.utils import load_config, EarlyStopping, save_checkpoint

cfg = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds = make_datasets(cfg)
train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, num_workers=cfg.data.num_workers)
val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

model = SolarUNet(cfg).to(device)
criterion = CombinedLoss(cfg)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = cfg.training.lr,
                              weight_decay= cfg.training.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.training.epochs
)
stopper = EarlyStopping(cfg.training.early_stopping_patience)

best_val = float("inf")
for epoch in range(1, cfg.training.epochs + 1):
    model.train()
    train_loss = 0.0
    for images, masks, meta in train_loader:
        images, masks, meta = images.to(device), masks.to(device), meta.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, meta), masks, meta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, meta in val_loader:
            images, masks, meta = images.to(device), masks.to(device), meta.to(device)
            val_loss += criterion(model(images, meta), masks, meta).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step()
    print(f"Epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")

    save_checkpoint(model, optimizer, epoch, val_loss,
                    f"{cfg.training.checkpoint_dir}/epoch_{epoch:03d}.pt")
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss,
                        f"{cfg.training.checkpoint_dir}/best.pt")
    
    if stopper(val_loss):
        print("Early stopping.")
        break