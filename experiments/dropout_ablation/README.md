# Technical Report: Dropout Ablation for E90ML

**Experiment date:** 1 August 2026

**Environment:** macOS on Apple Silicon, PyTorch 2.13.0 with MPS

## 1. Purpose

In the initial E90ML study, the validation loss was observed to be lower—and the validation classification score higher—than the corresponding training values. The current implementation reports F1-score as the classification metric. This ordering is opposite to the usual expectation that a model should achieve lower loss and higher classification performance on the data used for optimization than on unseen validation data.

The purpose of this experiment is to determine whether this inversion can be explained by Dropout. The learning curves obtained with the tuned Dropout rate are compared with curves obtained after setting only the Dropout rate to zero. Particular attention is paid to whether the validation loss and F1-score remain consistent, because validation loss controls both early stopping and selection of the saved model weights.

## 2. Method

### 2.1 Compared conditions

| Condition | Dropout rate | Configuration |
|---|---:|---|
| Dropout ON | 0.2513583424 | `param/usr/v1_mac_demo.yaml` |
| Dropout OFF | 0.0 | `experiments/dropout_ablation/config_dropout_off.yaml` |

The Dropout ON rate was selected by the current Optuna study. For the Dropout OFF run, `training.dropout_rate_override: 0.0` changed only the Dropout rate. Batch size, network structure, learning rate, training and validation samples, split seed, scaler procedure, weighted-BCE loss, and early-stopping rule were kept unchanged.

### 2.2 Evaluation modes

Training loss and F1-score are accumulated from the optimization batches while the model is in `model.train()` mode. Dropout is therefore active and BatchNorm uses the statistics of each mini-batch.

Validation loss and F1-score are evaluated at the end of each epoch using `model.eval()` and `torch.inference_mode()`. Dropout is disabled and BatchNorm uses its running statistics. Validation loss is the monitored quantity for early stopping, and the weights giving its minimum value are saved as the final model. This is the standard evaluation procedure; applying Dropout during validation would make the validation result stochastic and would not represent inference-time performance.

## 3. Learning-curve comparison and discussion

![Dropout ON/OFF learning-curve comparison](../../plots/train/ptep_demo_dropout_comparison.png)

The validation curves are qualitatively consistent between the two conditions. Validation loss remains within the same narrow range and follows a similar learning trend, while validation F1-score stays at a comparable level. Neither curve develops visible instability or a systematic shift when Dropout is removed. Validation loss therefore remains a stable criterion for early stopping and selection of the saved model weights. The same criterion and patience are used in both runs; differences in the selected and stopping epochs simply reflect their respective validation histories.

In contrast, the relationship between the training and validation curves changes clearly. With Dropout ON, training loss remains higher than validation loss throughout the learning curve, and training F1-score is almost always lower than validation F1-score. With Dropout OFF, this ordering is limited to the initial part of training; the training loss subsequently falls below the validation loss, and the training F1-score generally rises above the validation F1-score. Removing Dropout therefore restores the conventional train/validation ordering for most of the learning history.

The combination of stable validation curves and changing training curves indicates that the inversion originates primarily from how the training metrics are measured. With Dropout ON, those metrics are obtained from networks with randomly masked activations, whereas validation uses the complete inference network with Dropout disabled. The training data are consequently evaluated under a noisier and more difficult condition than the validation data. Once this training-side perturbation is removed, the persistent inversion also disappears.

A small inversion remains during the first few Dropout OFF epochs. This can arise because the training metrics are online averages collected while the model weights change throughout an epoch, whereas validation is evaluated once using the final end-of-epoch weights. BatchNorm also uses mini-batch statistics during training and running statistics during validation. These remaining differences can explain the short initial behavior, but not the persistent inversion observed with Dropout ON.

## 4. Conclusion

The learning-curve comparison shows that training-time Dropout explains the observed persistent inversion between training and validation metrics. The validation loss and F1-score remain at consistent levels when Dropout is removed, while the training curves move from the inverted ordering to the conventional ordering after the first few epochs. The change is therefore not driven by an unstable validation metric; it is driven primarily by measuring training performance with Dropout active.

The result does not indicate that Dropout was incorrectly applied to validation data. The code follows the standard procedure: Dropout is active during optimization with `model.train()` and disabled during validation with `model.eval()`. Because the validation loss remains stable and is evaluated without Dropout, it remains an appropriate quantity for early stopping and selection of the saved weights. The lower validation loss and higher validation F1-score relative to training can be explained by the training/evaluation mode difference without invoking data leakage or an error in validation.

This experiment isolates the explanation of the learning-curve shape. It does not determine whether Dropout should be removed from the final model or compare the best achievable performance after independently retuning the no-Dropout model.

## 5. Reproduction

Run the no-Dropout training and regenerate the comparison figure with:

```bash
bash experiments/dropout_ablation/run_experiment.sh
```
