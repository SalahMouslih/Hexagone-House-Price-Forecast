## House Prices Predictions

This project includes a command-line interface to run engines for exploratory data analysis and machine learning on real estate data.

### Installation

To use this project, clone the repository and install the required packages using pip:

```
git clone https://github.com/salahmouslih/data-challenge.git
cd data-challenge
pip install -r requirements.txt
```

### Usage
To use this project, run the main.py script with the desired command line arguments. The available options are:

- preprocess : preprocess the specified raw data file(s).
- ml : run the machine learning engine.
- eda : run the exploratory data analysis engine.

#### Pre-processing Engine
To preprocess the raw data, run the following command:

``` python main.py --preprocess path/to/raw/data.csv ```

You can also preprocess multiple files at once using wildcards:

``` python main.py --preprocess path/to/directory/*.csv ```

The preprocessed files will be stored in the `processed` directory.

#### Machine Learning Engine
To run the machine learning engine, use the following command:

``` python main.py --ml ```

This engine trains machine learning models on the preprocessed data and saves the `models` and evaluation metrics in the models directory.

#### Exploratory Data Analysis Engine

To run the exploratory data analysis engine, use the following command:


```python main.py --eda```

This engine generates exploratory data analysis visualizations and saves them in the `plots` directory.

#### Help

To see the available command-line arguments, run the following command:

``` python main.py --help``` 

## Licence
## Acknowledgments 
