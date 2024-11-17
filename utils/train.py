import torch 


# Separate train_step function
def train_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc_metric):
    """
    A single training step, computes loss, gradients, and updates weights.
    """
    model.train()

    x_batch_train = torch.tensor(x_batch_train, dtype=torch.float32)
    y_batch_train = torch.tensor(y_batch_train, dtype=torch.float32)

    optimizer.zero_grad()

    # Forward pass
    logits, _ = model(x_batch_train)

    # Calculate loss
    loss = loss_fn(logits.squeeze(), y_batch_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Update metrics
    train_acc_metric += (torch.round(torch.sigmoid(logits.squeeze())) == y_batch_train).float().mean()

    return loss.item(), train_acc_metric


# Separate val_step function
def val_step(model, x_batch_val, y_batch_val, loss_fn, val_acc_metric):
    """
    A single validation step, computes loss and accuracy.
    """
    model.eval()

    x_batch_val = torch.tensor(x_batch_val, dtype=torch.float32)
    y_batch_val = torch.tensor(y_batch_val, dtype=torch.float32)

    # Forward pass
    logits, _ = model(x_batch_val)

    # Calculate loss
    loss = loss_fn(logits.squeeze(), y_batch_val)

    # Update metrics
    val_acc_metric += (torch.round(torch.sigmoid(logits.squeeze())) == y_batch_val).float().mean()

    return loss.item(), val_acc_metric


def train(model, train_bags, fold, val_bags, args):
    """
    Train the Graph Att Net
    Parameters
    ----------
    train_bags: Data for training (list of patches)
    fold: Current fold in k-fold validation
    val_bags: Data for validation
    args: Arguments for the model, including hyperparameters and directories
    """
    # Data generators for training and validation
    train_gen = DataGenerator(args=args, batch_size=1, shuffle=False, filenames=train_bags, train=True)
    val_gen = DataGenerator(args=args, batch_size=1, shuffle=False, filenames=val_bags, train=True)

    # Create directories for saving models
    if not os.path.exists(os.path.join(args.save_dir, fold)):
        os.makedirs(os.path.join(args.save_dir, fold, args.experiment_name), exist_ok=True)

    checkpoint_path = os.path.join(os.path.join(args.save_dir, fold, args.experiment_name), "{}.pt".format(args.experiment_name))

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_fn = nn.BCEWithLogitsLoss() if not model.subtyping else nn.CrossEntropyLoss()

    # Early stopping parameters
    early_stopping = 20
    loss_history = deque(maxlen=early_stopping + 1)

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        print(f"\nStart of epoch {epoch}")
        start_time = time.time()

        # Training Step
        for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
            loss, train_acc_metric = train_step(model, x_batch_train, y_batch_train, optimizer, loss_fn, train_acc)
            train_loss += loss
            train_acc += train_acc_metric

        # Validation Step
        model.eval()
        with torch.no_grad():
            for step, (x_batch_val, y_batch_val) in enumerate(val_gen):
                loss, val_acc_metric = val_step(model, x_batch_val, y_batch_val, loss_fn, val_acc)
                val_loss += loss
                val_acc += val_acc_metric

        # Average loss and accuracy
        avg_train_loss = train_loss / len(train_gen)
        avg_train_acc = train_acc / len(train_gen)
        avg_val_loss = val_loss / len(val_gen)
        avg_val_acc = val_acc / len(val_gen)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Time: {time.time() - start_time:.2f}s")

        # Early Stopping & Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

        loss_history.append(avg_val_loss)
        if len(loss_history) > early_stopping and min(loss_history) == loss_history[0]:
            print(f'\nEarly stopping. No validation loss improvement in {early_stopping} epochs.')
            break


def predict(model, test_bags, fold, args):
    """
    Evaluate the model on the test set
    Parameters
    ----------
    test_bags: Data for testing
    fold: Current fold in k-fold validation
    args: Arguments for the model, including directories and hyperparameters
    """
    test_gen = DataGenerator(args=args, batch_size=1, filenames=test_bags, train=False)

    checkpoint_path = os.path.join(os.path.join(args.save_dir, fold, args.experiment_name), "{}.pt".format(args.experiment_name))
    model.load_state_dict(torch.load(checkpoint_path))

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for x_batch_val, y_batch_val in test_gen:
            x_batch_val = torch.tensor(x_batch_val, dtype=torch.float32)
            y_batch_val = torch.tensor(y_batch_val, dtype=torch.float32)

            logits, _ = model(x_batch_val)
            y_true.append(y_batch_val.numpy())
            y_pred.append(torch.sigmoid(logits.squeeze()).numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, np.round(y_pred), average='macro')
    recall = recall_score(y_true, np.round(y_pred), average='macro')
    f1 = f1_score(y_true, np.round(y_pred), average='macro')

    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return auc, precision, recall, f1
  