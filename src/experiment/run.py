import subprocess

seq_length = [32, 64, 128, 256]
hidden_size = [16, 32, 64]
num_hidden_layers = [2, 4, 8]
num_attention_heads = [2, 4, 8]
gpu = [0, 1, 2, 3, 4, 5, 6, 7]

# Run the first command
cmd1 = ['python', 'pretrain_bert.py',
        '--vocab_size=1024',
        '--modality=landmarks',
        '--experiment_id=UL01',
        '--log_mode=info',
        '--seq_length=128',
        '--hidden_size=128',
        '--num_hidden_layers=2',
        '--num_attention_heads=2',
        '--num_train_epochs=100',
        '--gpu=7']

subprocess.run(cmd1, check=True)

# Run the second command
cmd2 = ['python', 'finetune_bert.py',
        '--vocab_size=1024',
        '--log_mode=info',
        '--seq_length=128',
        '--modality=landmarks',
        '--experiment_id=UL01',
        '--hidden_size=128',
        '--num_tune_epochs=10',
        '--freeze_bert=False',
        '--batch_size=32',
        '--gpu=7']

subprocess.run(cmd2, check=True)
