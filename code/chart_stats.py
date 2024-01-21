from pathlib import Path
import matplotlib.pyplot as plt
import json

def convert_to_list_sorted_by_keys(stats: dict):
    stats_list = [(int(key), value[:2]) for key, value in stats.items()]

    return sorted(stats_list, key=lambda entry: entry[0])

def main():
    stats1 = json.loads(Path('tmp/stats_195k.json').read_text())
    stats2 = json.loads(Path('tmp/stats.json').read_text())

    stats1 = convert_to_list_sorted_by_keys(stats1)
    stats2 = convert_to_list_sorted_by_keys(stats2)

    stats1_last_batch_num = stats1[-1][0]
    batches = [entry[0] for entry in stats1]
    batches.extend([entry[0] + stats1_last_batch_num for entry in stats2])
    train_losses = [entry[1][0] for entry in stats1]
    train_losses.extend([entry[1][0] for entry in stats2])
    val_losses = [entry[1][1] for entry in stats1]
    val_losses.extend([entry[1][1] for entry in stats2])

    plt.plot(batches, train_losses, label='Training')
    plt.plot(batches, val_losses, label='Validation')
    plt.xlabel('n batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()