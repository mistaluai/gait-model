import torch
from sklearn.metrics import accuracy_score


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=10,
):
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for x, seq_lengths, y in train_loader: # we need to edit the get item to return the length of the sequence before padding
            
            x, seq_lengths, y = x.to(device), seq_lengths.to(device), y.to(device)

            optimizer.zero_grad()
            
            output = model(x, seq_lengths)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        model.eval()
        total_val_loss = 0.0
        
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, seq_lengths, y in val_loader:
            x, seq_lengths = x.to(device), seq_lengths.to(device)
            logits = model(x, seq_lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc    



from sklearn.model_selection import StratifiedKFold

def run_5fold_cv(sequences, labels, num_classes, epochs=20, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    sequences = np.array(sequences)
    labels = np.array(labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n--- Fold {fold + 1} ---")

        # train_data = GEIDataset(sequences[train_idx], labels[train_idx])
        # val_data = GEIDataset(sequences[val_idx], labels[val_idx])

        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # model = GEIConvLSTMClassifier(num_classes=num_classes)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        acc = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)
        fold_accuracies.append(acc)

    print(f"\nAverage Accuracy over 5 folds: {np.mean(fold_accuracies):.4f}")
    return fold_accuracies
