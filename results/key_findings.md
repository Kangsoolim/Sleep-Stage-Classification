# Key findings

- On the accepted OOF predictions (n = 9,913), Standard RF had Accuracy 0.7689 and Macro-F1 0.6894; Balanced RF had Accuracy 0.7664 and Macro-F1 0.7061.
- Across the five held-out subject folds, Accuracy was 0.7688 ± 0.0215 for Standard RF and 0.7661 ± 0.0289 for Balanced RF; fold-wise Macro-F1 was 0.6795 ± 0.0210 and 0.6998 ± 0.0255, respectively.
- With class weighting, N1 F1 changed from 0.2980 to 0.3565 (+0.0584), and REM F1 changed from 0.6933 to 0.7047 (+0.0114).
- The largest off-diagonal error for Standard RF was Stage 3+4 to N2 (343 epochs); Figure 4 provides the full count matrices for both models.
- The balanced setting has higher OOF Macro-F1 (+0.0168) alongside lower OOF Accuracy (-0.0025), an observed trade-off in this fixed comparison.
- Results use one 10-subject, recording-1, five-fold grouped evaluation and no statistical hypothesis test; observed differences are not interpreted as statistically significant or as evidence beyond this setting.
