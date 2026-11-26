"""Install official portable Blender inside this project only, if missing."""
from pathlib import Path
import urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
tools=ROOT/'.tools';exe=tools/'blender-4.5.3-windows-x64/blender.exe'
if not exe.exists():
 tools.mkdir(exist_ok=True);archive=tools/'blender-4.5.3-windows-x64.zip'
 print('Downloading official Blender 4.5.3 LTS portable package...')
 urllib.request.urlretrieve('https://download.blender.org/release/Blender4.5/blender-4.5.3-windows-x64.zip',archive)
 with zipfile.ZipFile(archive) as z:
  root=tools.resolve()
  if any(not (tools/n).resolve().is_relative_to(root) for n in z.namelist()):raise RuntimeError('Invalid archive member')
  z.extractall(tools)
print(exe)
