"""Fetch just the registered input from the published example bundle."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,tempfile
ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/tmdals1213-cyber/exercise-mechanics-lab/releases/download/v0.1.0/exercise-mechanics-github.zip'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',help='Use an already downloaded ZIP instead of the release URL');args=ap.parse_args();folder=ROOT/'examples/long_jump';meta=json.loads((folder/'prepared_model.json').read_text(encoding='utf-8'));target=folder/'registered_model.mjb.gz'
    def valid():return target.exists()and hashlib.sha256(target.read_bytes()).hexdigest()==meta['compressed_sha256']
    inverse_meta=json.loads((folder/'inverse_metadata.json').read_text(encoding='utf-8'));inverse=folder/'inverse_problem.npz'
    if valid()and inverse.exists()and hashlib.sha256(inverse.read_bytes()).hexdigest()==inverse_meta['fixture_sha256']:print('Registered mechanical and inverse inputs are ready.');return
    with tempfile.TemporaryDirectory()as tmp:
        archive=Path(args.archive)if args.archive else Path(tmp)/'example.zip'
        if not args.archive:urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive)as z:
            content=z.read('exercise-mechanics/examples/long_jump/registered_model.mjb.gz');inverse_data=z.read('exercise-mechanics/examples/long_jump/inverse_problem.npz')
        assert hashlib.sha256(inverse_data).hexdigest()==inverse_meta['fixture_sha256'],'Inverse input hash mismatch';inverse.write_bytes(inverse_data)
        assert hashlib.sha256(content).hexdigest()==meta['compressed_sha256'],'Example input hash mismatch';target.write_bytes(content)
    print('Registered example input downloaded and verified.')

if __name__=='__main__':main()
