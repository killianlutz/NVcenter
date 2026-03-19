### Getting started 
Open the terminal and run
```
git clone https://github.com/killianlutz/NVcenter.git
```
Create and activate a new virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

Install the required packages
```
pip install -r requirements.txt
```

Choose parameters inside `scripts/_config.py` and run the optimizer
```
python -m scripts.nvcenter
```

This creates or overwrites the file `./sims/example.npz` to store both the target gate, the gate time, the optimal control and the correpsonding pulses.

### Troubleshooting
Feel free to reach out to me: [Killian Lutz](https://killianlutz.github.io/).