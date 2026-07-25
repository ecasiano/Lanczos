# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Lanczos ED macOS app.

Build with:
    pyinstaller LanczosED.spec

Or use the build script:
    ./build_mac.sh
"""
import os
import sys

block_cipher = None

# Find the icon file
icon_path = os.path.join('icons', 'LanczosED.icns')
if not os.path.exists(icon_path):
    icon_path = None

a = Analysis(
    ['lanczos_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Numba and LLVM
        'numba',
        'numba.core',
        'numba.core.types',
        'numba.np.ufunc',
        'numba.typed',
        'llvmlite',
        'llvmlite.binding',
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
        'lanczos_ed.observables.tee',
        'lanczos_ed.gui',
        'lanczos_ed.gui.main_window',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        # Exclude Qt bindings we don't use (PyInstaller can't bundle two)
        'PyQt5',
        'PyQt6',
        'qtpy',
        # Heavy packages pulled in by conda but not needed
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

# Collect all numba/llvmlite files (templates, runtime support, etc.)
# Fix malformed TOC entries: collect_all can produce 2-tuples but
# PyInstaller 6.x COLLECT expects 3-tuples (dest, source, typecode).
from PyInstaller.utils.hooks import collect_all

def _fix_toc(entries, default_type='DATA'):
    """Ensure every TOC entry is a 3-tuple."""
    return [(e[0], e[1], e[2] if len(e) > 2 else default_type) for e in entries]

for pkg in ('numba', 'llvmlite'):
    datas, binaries, hiddenimports = collect_all(pkg)
    a.datas += _fix_toc(datas, 'DATA')
    a.binaries += _fix_toc(binaries, 'BINARY')
    a.hiddenimports += hiddenimports

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
    console=False,      # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS: support drag-and-drop
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
