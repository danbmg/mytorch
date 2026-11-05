# Training report: large

- `my_torch_network_final.nn`: 1024 relu > 512 relu > 5 softmax, 1319941 parameters, config `configs/large.conf`
- 71 epochs in 11517s on 2289784 boards, best epoch 65 (val_loss 0.1414), early stopping

![learning curves](curves.png)

## Test set (71557 boards never used for training or model selection)

Prediction averaged over the 4 symmetries of each board, as `my_torch_analyzer --predict` does (accuracy of the network on the boards alone: 0.9501).

```
                 precision  recall      f1  support
Nothing             0.9933  0.9759  0.9845    29757
Check White         0.9301  0.9351  0.9326    13418
Check Black         0.9371  0.9389  0.9380    11396
Checkmate White     0.9138  0.9369  0.9252     9276
Checkmate Black     0.9171  0.9401  0.9285     7710
accuracy                            0.9534    71557
macro f1                            0.9418

confusion matrix (rows: true class, columns: predicted)
                         0        1        2        3        4
Nothing              29040      370      267       47       33
Check White             98    12547        0      773        0
Check Black             74        0    10700        0      622
Checkmate White         12      573        0     8691        0
Checkmate Black         11        0      451        0     7248
```

![confusion matrix](confusion.png)

## Per subset

| subset | boards | accuracy | macro F1 | recall Nothing | recall Check White | recall Check Black | recall Checkmate White | recall Checkmate Black |
|---|---|---|---|---|---|---|---|---|
| 10_pieces | 21306 | 0.9686 | 0.9607 | 0.9794 | 0.9556 | 0.9529 | 0.9705 | 0.9663 |
| 20_pieces | 29999 | 0.9466 | 0.9400 | 0.9690 | 0.9188 | 0.9223 | 0.9510 | 0.9494 |
| many_pieces | 20252 | 0.9477 | 0.9119 | 0.9794 | 0.9367 | 0.9485 | 0.8503 | 0.8578 |
