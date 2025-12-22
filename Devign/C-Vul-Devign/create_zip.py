import zipfile
import os
from pathlib import Path

zip_path = 'devign_pipeline.zip'
source_dir = Path('devign_pipeline')

# Remove old zip if exists
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file_path in source_dir.rglob('*'):
        if file_path.is_file():
            # Use as_posix() for forward slashes (Unix-style)
            arcname = file_path.as_posix()
            zf.write(str(file_path), arcname)
            print(f'Added: {arcname}')

# Verify
with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    print(f'\nTotal files: {len(names)}')
    print('Sample paths:')
    for n in names[:5]:
        has_backslash = '\\' in n
        print(f'  {n} - backslash: {has_backslash}')

print('\nDone!')
