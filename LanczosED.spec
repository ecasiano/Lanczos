# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Lanczos ED macOS app.

Build with:
    pyinstaller --noconfirm LanczosED.spec
"""
import os

block_cipher = None

icon_path = os.path.join('icons', 'LanczosED.icns')
if not os.path.exists(icon_path):
    icon_path = None

# ---- collect numba/llvmlite without collect_all ----
# collect_all produces entries with absolute dest paths that break
# the macOS BUNDLE step.  Use targeted helpers instead.
from PyInstaller.utils.hooks import (
    collect_submodules, collect_dynamic_libs, collect_data_files,
)

extra_hiddenimports = (
    collect_submodules('numba')
    + collect_submodules('llvmlite')
)

extra_binaries = (
    collect_dynamic_libs('numba')
    + collect_dynamic_libs('llvmlite')
)

# Only collect data files that have relative dest paths
# (absolute paths crash the BUNDLE symlink logic)
_raw_datas = collect_data_files('numba') + collect_data_files('llvmlite')
extra_datas = []
for entry in _raw_datas:
    # entry = (source_path, dest_dir) — filter out absolute dests
    dest = entry[1] if len(entry) >= 2 else ''
    if os.path.isabs(dest):
        continue
    extra_datas.append(entry)

a = Analysis(
    ['lanczos_app.py'],
    pathex=['.'],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=[
        # SciPy sparse
        'scipy.sparse',
        'scipy.sparse.linalg',
        'scipy.sparse.linalg._eigen',
        'scipy.sparse.linalg._eigen.arpack',
        'scipy.sparse.csgraph',
        'scipy.linalg',
        # PySide6
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Our package
        'lanczos_ed',
        'lanczos_ed.basis',
        'lanczos_ed.unary_basis',
        'lanczos_ed.cli',
        'lanczos_ed.warmup',
        'lanczos_ed.symmetry',
        'lanczos_ed.symmetry_2d',
        'lanczos_ed.models',
        'lanczos_ed.models.bose_hubbard',
        'lanczos_ed.models.bose_hubbard_2d',
        'lanczos_ed.models.bose_hubbard_3d',
        'lanczos_ed.models.bose_hubbard_kagome',
        'lanczos_ed.models.fractional_chern',
        'lanczos_ed.solvers',
        'lanczos_ed.solvers.lanczos',
        'lanczos_ed.solvers.matrix_free',
        'lanczos_ed.observables',
        'lanczos_ed.observables.basic',
        'lanczos_ed.observables.ppee',
        'lanczos_ed.observables.tee',
        'lanczos_ed.gui',
        'lanczos_ed.gui.main_window',
    ] + extra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'PyQt5',
        'PyQt6',
        'qtpy',
        'pandas',
        'sqlalchemy',
        'tables',
        'zmq',
        'sqlite3',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LanczosED',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LanczosED',
)

app = BUNDLE(
    coll,
    name='Lanczos ED.app',
    icon=icon_path,
    bundle_identifier='com.ecasiano.lanczos-ed',
    info_plist={
        'CFBundleName': 'Lanczos ED',
        'CFBundleDisplayName': 'Lanczos ED',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '12.0',
    },
)
