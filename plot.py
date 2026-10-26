import argparse
import dataclasses
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from rosbags.highlevel import AnyReader

def flatten_msg(msg, prefix=''):
    """
    Recursively flattens a ROS 2 message object into a flat dictionary.
    Handles rosbags dataclasses, standard slots, and numpy arrays.
    """
    items = {}
    
    # 1. Handle rosbags dynamically generated dataclasses
    if dataclasses.is_dataclass(msg):
        for field in dataclasses.fields(msg):
            val = getattr(msg, field.name)
            new_key = f"{prefix}.{field.name}" if prefix else field.name
            items.update(flatten_msg(val, new_key))
            
    # 2. Handle standard rclpy messages (fallback)
    elif hasattr(msg, '__slots__'):
        for slot in msg.__slots__:
            val = getattr(msg, slot)
            new_key = f"{prefix}.{slot}" if prefix else slot
            items.update(flatten_msg(val, new_key))
            
    # 3. Handle standard arrays/lists
    elif isinstance(msg, (list, tuple)):
        for i, val in enumerate(msg):
            new_key = f"{prefix}[{i}]"
            items.update(flatten_msg(val, new_key))
            
    # 4. Handle numpy arrays natively without needing numpy imported
    elif hasattr(msg, 'shape') and hasattr(msg, 'tolist'):
        for i, val in enumerate(msg.tolist()):
            new_key = f"{prefix}[{i}]"
            items.update(flatten_msg(val, new_key))
            
    # 5. Base primitives
    else:
        items[prefix] = msg
        
    return items

def process_bag(bag_path):
    """
    Iterates through the ROS 2 bag, deserializes payloads automatically, 
    and organizes data into Pandas-ready dictionaries.
    """
    data_store = {}

    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            
            # Deserialize payload
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Flatten the nested message structure
            flat_msg = flatten_msg(msg)
            
            # Append hardware timestamp (nanoseconds to seconds)
            flat_msg['timestamp'] = timestamp * 1e-9 

            if topic not in data_store:
                data_store[topic] = []

            data_store[topic].append(flat_msg)

    return data_store

def plot_data(data_store):
    """
    Filters for numeric variables and plots them. 
    Each topic generates a figure with subplots for its variables.
    """
    for topic, records in data_store.items():
        df = pd.DataFrame(records)
        
        if 'timestamp' not in df.columns:
            continue

        # Set index to time
        df.set_index('timestamp', inplace=True)
        
        # Force conversion of typed scalars (e.g., numpy floats) to standard numerics
        df = df.apply(pd.to_numeric, errors='ignore')
        
        # Extract only columns that are officially numbers
        df_numeric = df.select_dtypes(include=['number'])

        if df_numeric.empty:
            print(f"Skipping {topic}: No numeric data extracted after flattening.")
            continue

        num_cols = len(df_numeric.columns)
        print(f"Plotting {num_cols} variables for topic: {topic}")
        
        # Scale figure size dynamically based on variable count
        fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(12, 2 * num_cols), sharex=True)
        fig.suptitle(f"Topic Data: {topic}", fontsize=14, fontweight='bold')

        if num_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, df_numeric.columns):
            ax.plot(df.index, df_numeric[col], label=col, linewidth=1.5)
            ax.set_ylabel(col.split('.')[-1], rotation=0, labelpad=40, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=8)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
    print("Rendering plots. Close the current window to proceed to the next topic.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot all numeric variables from a ROS 2 bag.")
    parser.add_argument("bag_path", type=str, help="Path to the ROS 2 bag directory (or .mcap file).")
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        print(f"Error: Path {bag_path} does not exist.")
        exit(1)

    print(f"Initializing data extraction sequence for: {bag_path}")
    extracted_data = process_bag(bag_path)
    
    print("Data extraction complete. Compiling visualization parameters...")
    plot_data(extracted_data)