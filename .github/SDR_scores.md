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

### Performance of v3 hdemucs_mmi model

Track 'Zeno - Signs' from MUSDB18-HQ test set

PyTorch inference (using v3-mmi default segment length + LSTM max length of 200):
```
vocals          ==> SDR:   8.328  SIR:  18.943  ISR:  16.097  SAR:   8.563
drums           ==> SDR:   9.284  SIR:  18.123  ISR:  16.230  SAR:  10.125
bass            ==> SDR:   3.612  SIR:  10.313  ISR:   6.958  SAR:   3.077
other           ==> SDR:   7.122  SIR:  11.391  ISR:  14.363  SAR:   7.910
```
PyTorch inference (using v4 7.8s segment length + LSTM max length of 336):
```
vocals          ==> SDR:   8.304  SIR:  18.916  ISR:  16.087  SAR:   8.557
drums           ==> SDR:   9.279  SIR:  18.149  ISR:  16.203  SAR:  10.109
bass            ==> SDR:   3.601  SIR:  10.350  ISR:   6.971  SAR:   3.076
other           ==> SDR:   7.123  SIR:  11.373  ISR:  14.373  SAR:   7.907
```
CPP inference (this codebase, `demucs_v3.cpp`):
```
vocals          ==> SDR:   8.331  SIR:  18.783  ISR:  16.027  SAR:   8.476
drums           ==> SDR:   7.546  SIR:  13.867  ISR:  13.572  SAR:   8.524
bass            ==> SDR:   1.737  SIR:   7.808  ISR:   3.990  SAR:   0.628
other           ==> SDR:   6.689  SIR:   9.831  ISR:  14.362  SAR:   8.026
```

### Performance of multi-threaded inference

Zeno - Signs, Demucs 4s multi-threaded using the same strategy used in <https://freemusicdemixer.com>.

Optimal performance: `export OMP_NUM_THREADS=4` + 4 threads via cli args for a total of 16 physical cores on my 5950X.

This should be identical in SDR but still worth testing since multi-threaded large waveform segmentation may still impact demixing quality:
```
vocals          ==> SDR:   8.317  SIR:  18.089  ISR:  15.887  SAR:   8.391
drums           ==> SDR:   9.987  SIR:  18.579  ISR:  16.997  SAR:  10.755
bass            ==> SDR:   4.039  SIR:  12.531  ISR:   6.822  SAR:   3.090
other           ==> SDR:   7.405  SIR:  11.246  ISR:  14.186  SAR:   8.099
```

Multi-threaded fine-tuned:
```
vocals          ==> SDR:   8.636  SIR:  18.914  ISR:  16.525  SAR:   8.678
drums           ==> SDR:  10.509  SIR:  19.882  ISR:  17.154  SAR:  11.095
bass            ==> SDR:   4.683  SIR:   9.565  ISR:   9.077  SAR:   4.806
other           ==> SDR:   7.374  SIR:  12.824  ISR:  12.938  SAR:   7.878
```

### Time measurements

Regular, big threads = 1, OMP threads = 16:
```
real    10m23.201s
user    29m42.190s
sys     4m17.248s
```

Fine-tuned, big threads = 1, OMP threads = 16: probably 4x the above, since it's just tautologically 4 Demucs models.

Mt, big threads = 4, OMP threads = 4 (4x4 = 16):
```
real    4m9.331s
user    18m59.731s
sys     3m28.465s
```

Ft Mt, big threads = 4, OMP threads = 4 (4x4 = 16):
```
real    16m30.252s
user    74m27.250s
sys     14m40.643s
```

Mt, big threads = 8, OMP threads = 16:
```
real    4m9.304s
user    43m21.830s
sys     10m15.712s
```
