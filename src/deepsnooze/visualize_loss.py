import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker
import argparse

def plot_losses(csv_path):

    df = pd.read_csv(csv_path)
    epoch_df = df.groupby('epoch')[['train_loss', 'val_loss']].mean().reset_index()
    epoch_df['epoch'] += 1

    melted_df = epoch_df.melt(
        id_vars='epoch', 
        value_vars=['train_loss', 'val_loss'], 
        var_name='Loss Type', 
        value_name='Loss Value'
    )

    melted_df['Loss Type'] = melted_df['Loss Type'].replace({
        'train_loss': 'Train Loss', 
        'val_loss': 'Validation Loss'
    })


    sns.set_theme(style="darkgrid")
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=melted_df, 
        x='epoch', 
        y='Loss Value', 
        hue='Loss Type', 
        marker='o',
        linewidth=2.5,
        markersize=7
    )

    plt.title('Training vs Validation Loss per Epoch', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    
    save_path = Path(csv_path).parent / "loss_curve.png"
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=20))

    plt.legend(title='', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

def get_latest_version(log_dir="lightning_logs"):
    """Scans the log directory and returns the path to the newest version's metrics.csv"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find directory: {log_dir}")
        
    
    versions = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith("version_")]
    
    if not versions:
        raise FileNotFoundError(f"No version folders found inside {log_dir}")
        
    latest_version = max(versions, key=lambda d: int(d.name.split('_')[1]))
    
    csv_path = latest_version / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Found {latest_version.name}, but no metrics.csv inside it!")
        
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyTorch Lightning training metrics.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default=None, 
        help="Specific path to a metrics.csv file. If left blank, uses the newest version."
    )
    
    args = parser.parse_args()
    
    try:
        if args.csv:
            target_csv = Path(args.csv)
            print(f"Using specified metrics file: {target_csv}")
        else:
            target_csv = get_latest_version()
            print(f"Auto-detected newest run: {target_csv.parent.name}")
            
        plot_losses(target_csv)
        
    except Exception as e:
        print(f"Error: {e}")