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

## Additional Notes
By default, this project uses the CPU-only version of PyTorch and DGL. However, if you have an NVIDIA graphics card in your machine, you can install the versions of PyTorch and DGL that match your CUDA version in order to run the model on your GPU.
