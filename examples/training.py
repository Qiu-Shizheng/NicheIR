"""Run from a directory where you want the synthetic training outputs saved."""
from nicheir.example import run_training_example

if __name__=='__main__':
    print(run_training_example('training_example',epochs=2))
