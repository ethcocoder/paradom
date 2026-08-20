# Google Colab Instructions for Finetuning Open Llama 3B with Adam Persona

This guide provides step-by-step instructions to set up and run the Open Llama 3B finetuning process for the "Adam" AI persona on Google Colab. This process uses the `v3` branch of the `ethcocoder/paradom` repository.

## 1. Open a New Google Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook (`File > New notebook`).

## 2. Set up GPU Runtime

Ensure you have a GPU runtime enabled for faster training:

*   Click on `Runtime` in the top menu.
*   Select `Change runtime type`.
*   Under `Hardware accelerator`, choose `GPU`.
*   Click `Save`.

## 3. Clone the Repository and Navigate to the Branch

Run the following commands in a Colab code cell to clone the `paradom` repository and switch to the `v3` branch:

```bash
!git clone https://github.com/ethcocoder/paradom.git
%cd paradom
!git checkout v3
```

## 4. Install Dependencies

Install all necessary Python libraries. This includes `torch`, `transformers`, `datasets`, `pandas`, `pyarrow`, and `tqdm`.

```bash
!pip install torch transformers datasets pandas pyarrow tqdm
```

## 5. Create the Adam Persona Dataset

Execute the `create_dataset.py` script to generate the `adam_alpaca.parquet` file, which contains the training data for the Adam persona.

```bash
!python3 create_dataset.py
```

This will output:

```
Created adam_alpaca.parquet with 30 rows.
```

## 6. Run the Finetuning Script

Now, run the `finetune_paradox.py` script. This script is configured to use the `openlm-research/open_llama_3b_v2` model and the `adam_alpaca.parquet` dataset. It includes a monitoring mechanism to observe loss between steps 1300 and 2000 to help prevent overfitting.

**Important Note**: The script is designed to run for a specified number of steps (`max_steps=2100`). You can stop the training manually at any point if you have verified it's running correctly, as per the original request. The script will save the model checkpoints to `./adam-finetuned`.

```bash
!python3 finetune_paradox.py
```

**Expected Output during Training (example):**

```
Loading data from adam_alpaca.parquet...
Loading model openlm-research/open_llama_3b_v2...
# ... (model and tokenizer loading messages) ...
Starting training...
# ... (training logs, including loss monitoring) ...

[MONITOR] Step 1300: Monitoring loss to prevent overfitting. Current Loss: X.XXXX
# ...
[MONITOR] Step 2000: Monitoring loss to prevent overfitting. Current Loss: Y.YYYY
# ...
```

Once you observe the training process starting and the loss monitoring messages, you can choose to stop the execution of the cell if you only need to verify the setup. The model will be saved in the `./adam-finetuned` directory.

## 7. Accessing the Finetuned Model

After the training (or verification run) is complete, the finetuned model and tokenizer will be saved in the `./adam-finetuned` directory within your Colab environment. You can then download these files or use them directly in Colab for inference.
