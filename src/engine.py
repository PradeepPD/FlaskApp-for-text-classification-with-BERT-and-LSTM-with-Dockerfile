import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(predictions, targets):
    return nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for bi, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        mask = dataset["mask"]
        target = dataset["target"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model,  device):
    model.eval()
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for bi, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = dataset["ids"]
            token_type_ids = dataset["token_type_ids"]
            mask = dataset["mask"]
            target = dataset["target"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            # loss = loss_fn(outputs, target)
            final_targets.extend(target.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return final_outputs, final_targets


def lstm_train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = []
    for _, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        x = dataset[0].to(device)
        y = dataset[1].to(device)

        # model.zero_grad()
        out = model(x)
        # print(out.shape)
        optimizer.zero_grad()
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        final_loss.append(loss.item())
    return np.mean(final_loss)


def lstm_eval_fn(data_loader, model, device):
    model.eval()
    final_targets = []
    final_outputs = []
    final_loss = []
    with torch.no_grad():
        for _, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            x = dataset[0].to(device)
            y = dataset[1].to(device)

            out = model(x)
            loss = loss_fn(out, y)
            final_targets.extend(y.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(out).cpu().detach().numpy().tolist())
            final_loss.append(loss.item())
    return final_outputs, final_targets, np.mean(final_loss)

