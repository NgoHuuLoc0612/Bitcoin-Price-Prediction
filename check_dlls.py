import os, ctypes

os.chdir(r'C:\Users\HP V\Downloads\Bitcoin Price Prediction')

dlls = [
    'btc_engine.cp311-win_amd64.pyd',
    'libbtc_core.dll',
    'libopenblas.dll',
    'libgfortran-5.dll',
    'libgomp-1.dll',
    'libgcc_s_seh-1.dll',
    'libstdc++-6.dll',
    'libwinpthread-1.dll',
    'libquadmath-0.dll',
]

for dll in dlls:
    try:
        ctypes.CDLL(os.path.abspath(dll))
        print(f'OK: {dll}')
    except OSError as e:
        print(f'FAIL: {dll} -> {e}')
