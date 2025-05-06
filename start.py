import argparse
import tools.train as train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model_name', type=str, help='A string argument')
    parser.add_argument('--seq_len', type=int, help='An integer argument')
    
    args = parser.parse_args()
    train(args)