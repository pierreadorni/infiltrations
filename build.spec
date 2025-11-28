# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for multi-platform distribution (onefile, noconsole)

import sys
import os

block_cipher = None

# Determine console and icon based on platform
is_windows = sys.platform == 'win32'
is_macos = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

console = False  # noconsole for all platforms
icon_path = None
datas = []

# Platform-specific configurations
if is_macos:
    icon_path = 'cell.icns'
elif is_windows:
    icon_path = 'cell.ico'
elif is_linux:
    icon_path = None  # Linux doesn't use icon in spec for windowed apps

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Cell Infiltrations',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=console,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if icon_path and os.path.exists(icon_path) else None,
    onefile=True,
)
