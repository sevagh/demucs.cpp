# SDR scores

### Performance of 4-source model

Track 'Zeno - Signs' from MUSDB18-HQ test set

PyTorch custom inference in [my script](./scripts/demucs_pytorch_inference.py):
```
vocals          ==> SDR:   8.339  SIR:  18.274  ISR:  15.835  SAR:   8.354
drums           ==> SDR:  10.058  SIR:  18.598  ISR:  17.023  SAR:  10.812
bass            ==> SDR:   3.926  SIR:  12.414  ISR:   6.941  SAR:   3.202
other           ==> SDR:   7.421  SIR:  11.289  ISR:  14.241  SAR:   8.179
```
CPP inference (this codebase):
```
vocals          ==> SDR:   8.339  SIR:  18.276  ISR:  15.836  SAR:   8.346
drums           ==> SDR:  10.058  SIR:  18.596  ISR:  17.019  SAR:  10.810
bass            ==> SDR:   3.919  SIR:  12.436  ISR:   6.931  SAR:   3.182
other           ==> SDR:   7.421  SIR:  11.286  ISR:  14.252  SAR:   8.183
```
*n.b.* for the above results, the random shift in the beginning of the song was fixed to 1337 in both PyTorch and C++.

### Performance of 6-source model

Track 'Zeno - Signs' from MUSDB18-HQ test set

PyTorch custom inference in [my script](./scripts/demucs_pytorch_inference.py) with `--six-source` flag:
```
vocals          ==> SDR:   8.396  SIR:  18.695  ISR:  16.076  SAR:   8.580
drums           ==> SDR:   9.928  SIR:  17.930  ISR:  17.523  SAR:  10.635
bass            ==> SDR:   4.522  SIR:  10.447  ISR:   8.618  SAR:   4.374
other           ==> SDR:   0.168  SIR:  11.449  ISR:   0.411  SAR:  -2.720
```
CPP inference (this codebase):
```
vocals          ==> SDR:   8.395  SIR:  18.699  ISR:  16.076  SAR:   8.576
drums           ==> SDR:   9.927  SIR:  17.921  ISR:  17.518  SAR:  10.635
bass            ==> SDR:   4.519  SIR:  10.458  ISR:   8.606  SAR:   4.370
other           ==> SDR:   0.164  SIR:  11.443  ISR:   0.409  SAR:  -2.713
```

*n.b.* the "other" score will be artificially low because of the extra guitar + piano separation where there are no stems to compare to

### Performance of fine-tuned/ft model

Track 'Zeno - Signs' from MUSDB18-HQ test set

PyTorch custom inference in [my script](./scripts/demucs_pytorch_inference.py) with `--fine-tuned` flag:
```
vocals          ==> SDR:   8.630  SIR:  18.962  ISR:  16.419  SAR:   8.701
drums           ==> SDR:  10.478  SIR:  19.880  ISR:  17.127  SAR:  11.128
bass            ==> SDR:   4.468  SIR:   9.477  ISR:   9.060  SAR:   4.895
other           ==> SDR:   7.384  SIR:  12.812  ISR:  12.977  SAR:   7.798
```
CPP inference (this codebase, `demucs_ft.cpp`)
```
vocals          ==> SDR:   8.594  SIR:  19.045  ISR:  16.313  SAR:   8.617
drums           ==> SDR:  10.463  SIR:  19.782  ISR:  17.144  SAR:  11.132
bass            ==> SDR:   4.584  SIR:   9.359  ISR:   9.068  SAR:   4.885
other           ==> SDR:   7.426  SIR:  12.793  ISR:  12.975  SAR:   7.830
```

### Performance of multi-threaded inference

Zeno - Signs, Demucs 4s multi-threaded. This should be identical in SDR but still worth testing since multi-threaded large waveform segmentation may still impact demixing quality:
```
```

Same strategy used by <https://freemusicdemixer.com>.
