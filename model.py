import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class ColaModel(pl.LightningModule):
    #google//bert_uncased_L-2_H-128_A-2
    #google-bert/bert-base-uncased
    def __init__(self, model_name="google//bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        # self.bert = AutoModel.from_pretrained(model_name)
        # self.W = nn.Linear(self.bert.config.hidden_size, 2)
        
        #week 2
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=self.num_classes)
        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            task="multiclass",average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            task="multiclass",average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(task="multiclass",num_classes=self.num_classes,average="micro")
        self.recall_micro_metric = torchmetrics.Recall(task="multiclass",num_classes=self.num_classes,average="micro")

        self.validation_step_outputs = []
        
    def on_fit_start(self):
        """Called when fit begins"""
        # Move metrics to the device where the model is
        device = self.device
        self.train_accuracy_metric = self.train_accuracy_metric.to(device)
        self.val_accuracy_metric = self.val_accuracy_metric.to(device)
        self.f1_metric = self.f1_metric.to(device)
        self.precision_macro_metric = self.precision_macro_metric.to(device)
        self.recall_macro_metric = self.recall_macro_metric.to(device)
        self.precision_micro_metric = self.precision_micro_metric.to(device)
        self.recall_micro_metric = self.recall_micro_metric.to(device)
        
    def forward(self, input_ids, attention_mask,labels=None):
        # tensorboard 
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # h_cls = outputs.last_hidden_state[:, 0]
        # logits = self.W(h_cls)
        # return logits
        
        
        # outputs = self.bert(
        #     input_ids=input_ids, attention_mask=attention_mask, labels=labels
        # )
        # return outputs

        # WandB and AutoModel for Sequence Classification
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        # logits = self.forward(batch["input_ids"], batch["attention_mask"])
        # loss = F.cross_entropy(logits, batch["label"])
        # self.log("train_loss", loss, prog_bar=True)
        # return loss
        # week 2
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        # logits = self.forward(batch["input_ids"], batch["attention_mask"])
        # loss = F.cross_entropy(logits, batch["label"])
        # _, preds = torch.max(logits, dim=1)
        # val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        # val_acc = torch.tensor(val_acc)
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", val_acc, prog_bar=True)
        
        # week 2
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        
        self.validation_step_outputs.append({"labels": labels, "logits": outputs.logits})

        return {"labels": labels, "logits": outputs.logits}

    def on_validation_epoch_end(self):
        # Replace validation_epoch_end with on_validation_epoch_end
        outputs = self.validation_step_outputs
        
        # Clear the outputs list for the next epoch
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        # Wandb logging
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )
        
        # Reset the outputs list for next epoch
        self.validation_step_outputs.clear()
    
    # def validation_epoch_end(self, outputs):
    #     labels = torch.cat([x["labels"] for x in outputs])
    #     logits = torch.cat([x["logits"] for x in outputs])
    #     preds = torch.argmax(logits, 1)

    #     ## There are multiple ways to track the metrics
    #     # 1. Confusion matrix plotting using inbuilt W&B method
    #     self.logger.experiment.log(
    #         {
    #             "conf": wandb.plot.confusion_matrix(
    #                 probs=logits.numpy(), y_true=labels.numpy()
    #             )
    #         }
    #     )
        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    #week 1
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])