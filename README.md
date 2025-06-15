# phi2 Japanese NPC Fine-Tuning Project

This project is designed to fine-tune the phi2 model for use as NPCs (Non-Playable Characters) in games, specifically targeting Japanese language interactions. The goal is to create a model that can understand and generate contextually relevant dialogues for NPCs, enhancing the gaming experience.

## Project Structure

- **data/**: Contains raw and processed data files.
  - **raw/**: Raw data files used for training.
  - **processed/**: Processed data files ready for training.
  
- **notebooks/**: Jupyter notebooks for data preparation.
  - **data_preparation.ipynb**: Notebook for data cleaning and transformation.

- **src/**: Source code for training and evaluation.
  - **train.py**: Script for fine-tuning the phi2 model.
  - **evaluate.py**: Script for evaluating the fine-tuned model.
  - **export_onnx.py**: Script for exporting the model to ONNX format.
  - **utils.py**: Utility functions for data loading and preprocessing.

- **scripts/**: Shell scripts for automation.
  - **preprocess.sh**: Script for automating data preprocessing.
  - **run_training.sh**: Script for executing the training process.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **config.yaml**: Configuration settings for the training process.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd phi2-japanese-npc-finetune
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place your raw data files in the `data/raw/` directory.
   - Run the data preparation notebook to clean and transform the data.

4. Configure the training parameters in `config.yaml` as needed.

5. Run the preprocessing script:
   ```
   bash scripts/preprocess.sh
   ```

6. Start the training process:
   ```
   bash scripts/run_training.sh
   ```

## Usage

After training, you can evaluate the model using the `evaluate.py` script. To export the model for use in Unity, run the `export_onnx.py` script.

## Fine-Tuning Process

The fine-tuning process involves adjusting the pre-trained phi2 model on a dataset of Japanese dialogues tailored for NPC interactions. This includes:
- Data cleaning and transformation to ensure quality input.
- Training the model with appropriate hyperparameters.
- Evaluating the model's performance and making necessary adjustments.

By following this guide, you will be able to set up and fine-tune the phi2 model for your NPC needs in Japanese language games.