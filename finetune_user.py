"""
Fine-tuning routine for Graves RNN Handwriting Synthesis on user handwriting data.
Loads pretrained checkpoint-17900 and fine-tunes on processed user dataset.
"""

import os
import tensorflow as tf
from rnn import rnn, DataReader


def finetune_user_handwriting(
    data_dir: str = 'data/processed/',
    checkpoint_dir: str = 'checkpoints',
    warm_start_step: int = 17900,
    finetune_steps: int = 2500,
    learning_rate: float = 0.00005,
    batch_size: int = 16
):
    """
    Fine-tunes the pretrained RNN on the user's handwritten pairs.
    """
    assert os.path.exists(os.path.join(data_dir, 'x.npy')), f"Missing data in {data_dir}"

    print(f"[INFO] Initializing DataReader from {data_dir}...")
    dr = DataReader(data_dir=data_dir)

    print(f"[INFO] Setting up model from warm_start_step {warm_start_step}...")
    nn = rnn(
        reader=dr,
        log_dir='logs',
        checkpoint_dir=checkpoint_dir,
        prediction_dir='predictions',
        learning_rates=[learning_rate],
        batch_sizes=[batch_size],
        patiences=[500],
        beta1_decays=[0.9],
        validation_batch_size=min(batch_size, len(dr.val_df) if len(dr.val_df) > 0 else 1),
        optimizer='rms',
        num_training_steps=warm_start_step + finetune_steps,
        warm_start_init_step=warm_start_step,
        regularization_constant=0.0001,  # Weight regularization to avoid catastrophic forgetting
        keep_prob=0.9,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=200,
        log_interval=20,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )

    print(f"[INFO] Starting fine-tuning for {finetune_steps} steps...")
    nn.fit()
    print("[OK] Fine-tuning completed!")


if __name__ == '__main__':
    finetune_user_handwriting()
