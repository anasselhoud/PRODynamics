# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main-live.py'],
    pathex=["C:/Users/elhoudan/OneDrive - Faurecia/PhD Codes/PRODynamics"],
    binaries=[],
    datas=[('./config.yaml', '.'), ('./LineData.xlsx', '.'), ('./assets/icons/*.png', './assets/icons/'), ('./assets/icons/*.ico', './assets/icons/')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='prodynamics-spec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="C:/Users/elhoudan/OneDrive - Faurecia/PhD Codes/PRODynamics/assets/icons/mainicon2.ico"
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='prodynamics-spec',
)
