```
python3 -m ensurepip
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
cp env.sh.sample env.sh # fill in api keys
./index.sh YOUR_DOCUMENT.txt
./run.sh
```
