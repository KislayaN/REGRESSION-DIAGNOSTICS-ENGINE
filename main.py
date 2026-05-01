import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from core.analyze import Analyzer

def main():
    analyzer = Analyzer(show_plots=True)
     
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(as_frame=True)
    target = housing.frame.columns.to_list()[-1]
    data = housing.frame
    
    result = analyzer.run(data, target)
    return result

if __name__ == '__main__':
    results = main()