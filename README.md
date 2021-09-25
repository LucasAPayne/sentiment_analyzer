# Sentiment Analyzer
This project uses [Graph4NLP](https://github.com/graph4ai/graph4nlp) to construct a sentiment analyzer GNN model that predicts whether reviews on Rotten Tomatoes are positive or negative.

## Getting Started
### Installation
First, navigate to the desired folder on your system and recursively clone the repository along with its submodules:

    git clone --recursive https://github.com/LucasAPayne/sentiment_analyzer.git

It is recommended to use [Anaconda](https://www.anaconda.com/products/individual) to manage virtual environments. In an Anaconda command prompt, create and activate an environment for this project, and install its dependencies:

    conda create --name sentiment_analyzer
    conda activate sentiment_analyzer
    cd path_to_repo
    pip install -r requirements.txt

This project relies on [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/#download) to process data. It will need to be downloaded separately. Ensure that a 64-bit version of [Java 8+](https://www.java.com/en/download/manual.jsp) is installed on your machine. Java is necessary to run StanfordCoreNLP, and the 64-bit version is needed to allocate enough memory to it.

Also, this project requires Google Chrome version 93.0.4577.63, as it relies on that version of Chrome to gather the dataset from Rotten Tomatoes with Chromedriver. You can check your Chrome version in the About Chrome page in the Chrome settings. Currently, only the Windows version of Chromedriver is supported in this project.

### Running the Code
It is recommended to open JupyterLab from the repository's root directory so that all of the repo's folders are accessible through it. The notebook is in the `src/graph4nlp` directory. To open JupyterLab, use this command (in the Anaconda command prompt):

    jupyter-lab

If no processed data is available (if there is no `data/processed` directory from the root directory of the repository or it is empty), open a separate command prompt and run these commands to initialize an instance of StanfordCoreNLP to process the data:

    cd path_to_stanfordcorenlp
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

Now, the code should be ready to run from JupyterLab.

### Running on a GPU
By default, this project uses the CPU-only version of PyTorch and DGL. However, if you have an NVIDIA GPU and would like to train this model on it, all you need to do is install the versions of [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html) that correspond to your CUDA version.

## Results
| Dataset         | GNN Model | Graph Construction            | Accuracy |
| ---             | :---:     | :---:                         | :---:    |
| IMDb            | GraphSAGE | Constituency <br/> Dependency | 83.1% <br/> 82.9% |
| Rotten Tomatoes | GraphSAGE | Constituency <br/> Dependency | 69.5% <br/> 62.2% |

## Notes
### Datasets
The IMDb dataset used is a trimmed-down version of the one provided in the [torchtext](https://pytorch.org/text/stable/index.html) library. The original version contains 25,000 reviews in the train set and 25,000 reviews in the test set. This version keeps only 5,000 reviews from the train set and 2,000 reviews from the test set (1,000 of those reviews are used for this project's test set, and the other 1,000 are used for the validation set). Additionally, each review was cut off after the first period after 500 characters (to limit each review to roughly 100 words). This cut was made because of performance issues encountered with StanfordCoreNLP.

The Rotten Tomatoes dataset was originally developed to test the model in a real-world environment. The idea was for the model to evaluate all of the available reviews for a film and calculate its freshness score. However, the model was also trained on this dataset, where a subset of the audience reviews were used for training, and the full set of critic reviews were used for testing and validation. This dataset leads to severe overfitting, which is likely due to the low number of data points, the differences between audience and critic reviews, and the fact that all reviews concern the same film.

### StanfordCoreNLP Performance Issues
As mentioned above, the IMDb dataset was drastically reduced in this project due to limitations of StanfordCoreNLP. When attempting to process the entire dataset, StanfordCoreNLP would throw an OutOfMemoryError. This would occur no matter the amount of memory given to the process (it was attempted with 4GB, 8GB, and even 16GB of memory). Occasionally, this error would not be thrown, and the process would hang. It could be terminated and resumed to carry on processing. Completing the processing of the entire dataset in this way took several hours. However, reducing the size of the dataset to 7,000 total reviews and limiting the length of each one allowed StanfordCoreNLP to process the data without issues. The accuracy of the model would likely increase if the entire dataset could be efficiently processed.
