
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)
print(data.head())


def extract_features(data):
    data['vm'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['vm_mean'] = data['vm'].rolling(window=10, min_periods=1).mean()
    data['vm_std'] = data['vm'].rolling(window=10, min_periods=1).std()
    data['vm_var'] = data['vm'].rolling(window=10, min_periods=1).var()
    fft_result = np.abs(np.fft.fft(data['vm']))
    fft_summary = {
        'fft_mean': fft_result.mean(),
        'fft_max': fft_result.max()
    }
    data['fft_mean'] = fft_summary['fft_mean']
    data['fft_max'] = fft_summary['fft_max']
    data.dropna(inplace=True)

    return data

def train_or_load_model():
    try:
        # Load the pre-trained model
        model = joblib.load('sleep_stage_model.pkl')
        print("Loaded pre-trained model.")
    except:
        print("Training a new model...")
        # Example: Create synthetic training data
        np.random.seed(42)
        train_data = pd.DataFrame({
            'vm_mean': np.random.uniform(0, 50, 1000),
            'vm_std': np.random.uniform(0, 20, 1000),
            'vm_var': np.random.uniform(0, 10, 1000),
            'label': np.random.choice(['Awake', 'Light Sleep', 'Deep Sleep'], 1000)
        })

        # Preprocessing
        X = train_data[['vm_mean', 'vm_std', 'vm_var']]
        y = train_data['label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # Save the model
        joblib.dump(model, 'sleep_stage_model.pkl')

    return model
def analyze_sleep(file_path, model):

    # Load data
    data = pd.read_csv(file_path)
    data['time'] = pd.to_datetime(data['time'])

    # Feature engineering
    data = extract_features(data)

    # Select features for prediction
    features = data[['vm_mean', 'vm_std', 'vm_var']]

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Predict sleep stages
    data['stage'] = model.predict(features_scaled)

    # Calculate sleep metrics
    data['time_diff'] = data['time'].diff().dt.total_seconds().fillna(0)
    total_sleep_time = data.loc[data['stage'] != 'Awake', 'time_diff'].sum()
    awake_time = data.loc[data['stage'] == 'Awake', 'time_diff'].sum()
    deep_sleep_time = data.loc[data['stage'] == 'Deep Sleep', 'time_diff'].sum()
    light_sleep_time = data.loc[data['stage'] == 'Light Sleep', 'time_diff'].sum()

    return {
        'Total Sleep Time (seconds)': total_sleep_time,
        'Awake Time (seconds)': awake_time,
        'Deep Sleep Time (seconds)': deep_sleep_time,
        'Light Sleep Time (seconds)': light_sleep_time,
        'Predicted Data': data[['time', 'stage']]
    }

# Main program
file_path = "/content/accelerometer_datanew2.csv"  # Replace with your file path
model = train_or_load_model()  # Train or load the model
results = analyze_sleep(file_path, model)
def seconds_to_hrs_mins(seconds):
    hrs = int(seconds // 3600)  # Get hours
    mins = int((seconds % 3600) // 60)  # Get remaining minutes
    return f"{hrs} hrs {mins} mins"

# Recalculate Total Sleep Time
results['Total Sleep Time (seconds)'] = (
    results['Awake Time (seconds)'] +
    results['Deep Sleep Time (seconds)'] +
    results['Light Sleep Time (seconds)']
)

# Prepare results in hrs and mins
results_in_hrs_mins = {
    'Total Sleep Time': seconds_to_hrs_mins(results['Total Sleep Time (seconds)']),
    'Awake Time': seconds_to_hrs_mins(results['Awake Time (seconds)']),
    'Deep Sleep Time': seconds_to_hrs_mins(results['Deep Sleep Time (seconds)']),
    'Light Sleep Time': seconds_to_hrs_mins(results['Light Sleep Time (seconds)']),
}

# Print the results
for key, value in results_in_hrs_mins.items():
    print(f"{key}: {value}")

# Calculate Total Time in Bed (TIB)
start_time = results['Predicted Data']['time'].iloc[0]
end_time = results['Predicted Data']['time'].iloc[-1]
tib_seconds = (end_time - start_time).total_seconds()  # Total time in bed in seconds

# Calculate Sleep Efficiency (SE)
tst_seconds = results['Deep Sleep Time (seconds)'] + results['Light Sleep Time (seconds)']
sleep_efficiency = (tst_seconds / tib_seconds) * 100  # Sleep Efficiency as a percentage

# Add Sleep Efficiency to the results
results_in_hrs_mins['Sleep Efficiency (%)'] = f"{sleep_efficiency:.2f}%"

# Display updated results
print("\nUpdated Results with Sleep Efficiency:")
for key, value in results_in_hrs_mins.items():
    print(f"{key}: {value}")

"""**Visualize Data**

Plot Sleep Stages Over Time
"""

# Convert results to hours for plotting
time_data = {
    'Awake Time': results['Awake Time (seconds)'] / 3600,
    'Deep Sleep Time': results['Deep Sleep Time (seconds)'] / 3600,
    'Light Sleep Time': results['Light Sleep Time (seconds)'] / 3600,
    'Total Sleep Time': results['Total Sleep Time (seconds)'] / 3600
}

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

def generate_realistic_sleep_data(start_time, duration_hours=8):
    """Generate realistic sleep stage data"""
    # Create time range
    time_range = pd.date_range(
        start=start_time,
        end=start_time + timedelta(hours=duration_hours),
        freq='1min'
    )

    # Initialize empty data
    data = pd.DataFrame({
        'time': time_range,
        'stage': 'Awake'
    })

    # Simulate realistic sleep patterns
    # Typically: Awake -> Light -> Deep -> Light -> REM (90-min cycles)
    current_time = start_time

    # Initial falling asleep period (15-20 minutes)
    fall_asleep_idx = (data['time'] < (start_time + timedelta(minutes=20)))
    data.loc[fall_asleep_idx, 'stage'] = 'Awake'

    # Generate 90-minute sleep cycles
    cycle_duration = timedelta(minutes=90)
    num_cycles = int((duration_hours * 60) / 90)

    for cycle in range(num_cycles):
        cycle_start = start_time + timedelta(minutes=20) + (cycle * cycle_duration)

        # Light Sleep (30-40 minutes)
        light_sleep_mask = (data['time'] >= cycle_start) & \
                          (data['time'] < (cycle_start + timedelta(minutes=35)))
        data.loc[light_sleep_mask, 'stage'] = 'Light Sleep'

        # Deep Sleep (20-30 minutes)
        deep_sleep_mask = (data['time'] >= (cycle_start + timedelta(minutes=35))) & \
                         (data['time'] < (cycle_start + timedelta(minutes=60)))
        data.loc[deep_sleep_mask, 'stage'] = 'Deep Sleep'

        # Back to Light Sleep (30 minutes)
        light_sleep_2_mask = (data['time'] >= (cycle_start + timedelta(minutes=60))) & \
                            (data['time'] < (cycle_start + timedelta(minutes=90)))
        data.loc[light_sleep_2_mask, 'stage'] = 'Light Sleep'

        # Add some brief awakenings (1-3 minutes) randomly
        if np.random.random() < 0.3:  # 30% chance of brief awakening
            awakening_start = cycle_start + timedelta(minutes=np.random.randint(0, 90))
            brief_awakening_mask = (data['time'] >= awakening_start) & \
                                 (data['time'] < (awakening_start + timedelta(minutes=2)))
            data.loc[brief_awakening_mask, 'stage'] = 'Awake'

    return data

def plot_improved_sleep_analysis(data):
    """Create improved sleep visualizations"""
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2)

    # 1. Sleep Stage Hypnogram
    ax1 = fig.add_subplot(gs[0, :])
    stage_map = {'Awake': 2, 'Light Sleep': 1, 'Deep Sleep': 0}
    stage_numbers = data['stage'].map(stage_map)

    # Plot with steps to show distinct transitions
    ax1.step(data['time'], stage_numbers, 'b-', linewidth=1, where='post')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Deep', 'Light', 'Awake'])
    ax1.set_title('Sleep Stage Hypnogram')
    ax1.grid(True)
    ax1.set_xlim(data['time'].iloc[0], data['time'].iloc[-1])

    # Format x-axis to show hours
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # 2. Sleep Stage Distribution Pie Chart
    ax2 = fig.add_subplot(gs[1, 0])
    stage_dist = data['stage'].value_counts()
    colors = {'Awake': 'red', 'Light Sleep': 'yellow', 'Deep Sleep': 'blue'}
    stage_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=[colors[x] for x in stage_dist.index])
    ax2.set_title('Sleep Stage Distribution')

    # 3. Sleep Stage Timeline
    ax3 = fig.add_subplot(gs[1, 1])
    for stage, color in colors.items():
        mask = data['stage'] == stage
        if mask.any():
            ax3.fill_between(data['time'][mask], 0, 1, label=stage, color=color, alpha=0.5)

    ax3.set_title('Sleep Stage Timeline')
    ax3.legend()
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax3.set_xlim(data['time'].iloc[0], data['time'].iloc[-1])

    plt.tight_layout()
    return fig

# Example usage
start_time = datetime(2024, 11, 17, 23, 0)  # 11:00 PM
sleep_data = generate_realistic_sleep_data(start_time, duration_hours=8)

# Calculate metrics
def calculate_sleep_metrics(data):
    """Calculate sleep quality metrics"""
    metrics = {}

    total_duration = (data['time'].max() - data['time'].min()).total_seconds() / 3600
    metrics['Total Sleep Duration'] = f"{total_duration:.1f} hours"

    # Calculate percentages
    for stage in ['Awake', 'Light Sleep', 'Deep Sleep']:
        percentage = (data['stage'] == stage).mean() * 100
        metrics[f'{stage} Percentage'] = f"{percentage:.1f}%"

    # Calculate number of awakenings
    stage_changes = data['stage'] != data['stage'].shift()
    awakenings = sum((data['stage'] == 'Awake') & stage_changes)
    metrics['Number of Awakenings'] = awakenings

    return metrics

# Print metrics
metrics = calculate_sleep_metrics(sleep_data)
print("\nSleep Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")



def calculate_sleep_fragmentation_index(data):
    # Count stage transitions
    stage_transitions = (data['stage'] != data['stage'].shift()).sum()

    # Count brief awakenings (1-3 minutes)
    def identify_brief_awakenings(group):
        if len(group) <= 3 and group['stage'].iloc[0] == 'Awake':
            return 1
        return 0

    data['awakening_group'] = (data['stage'] != data['stage'].shift()).cumsum()
    brief_awakenings = data.groupby('awakening_group').apply(identify_brief_awakenings).sum()

    total_sleep_time = len(data[data['stage'] != 'Awake']) / 60  # in hours
    fragmentation_index = (stage_transitions + brief_awakenings) / total_sleep_time

    return fragmentation_index/10, stage_transitions, brief_awakenings

def calculate_comprehensive_sleep_metrics(data):
    metrics = {}

    # Sleep Fragmentation
    fragmentation, stage_transitions, brief_awakenings = calculate_sleep_fragmentation_index(data)
    metrics['Fragmentation Index'] = f"{fragmentation:.1f}"


    # Create a formatted display of metrics
    print("\nComprehensive Sleep Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return metrics
comprehensive_metrics = calculate_comprehensive_sleep_metrics(results['Predicted Data'])

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Explanation Table
def display_sleep_explanations():
    """
    Display an explanation table for sleep stages and their benefits.
    """
    html_table = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    <table>
        <tr>
            <th>Sleep Stage</th>
            <th>Description</th>
            <th>Real-Life Benefits</th>
        </tr>
        <tr>
            <td>Awake</td>
            <td>Periods of wakefulness; can include brief awakenings during sleep.</td>
            <td>Helps in conscious thought processing and responding to the environment.</td>
        </tr>
        <tr>
            <td>Light Sleep</td>
            <td>Transition stage where the body starts relaxing; majority of sleep time.</td>
            <td>Promotes relaxation and prepares the body for deeper stages of sleep.</td>
        </tr>
        <tr>
            <td>Deep Sleep</td>
            <td>The most restorative stage; occurs in longer chunks early in the night.</td>
            <td>Supports muscle repair, immune function, and memory consolidation.</td>
        </tr>
    </table>
    """
    display(HTML(html_table))

# Main Execution
display_sleep_explanations()


# Simulated Sleep Data
def generate_sleep_data():
    """
    Generate default sleep stage data for an 8-hour period.
    """
    stages = (
        ["Awake"] * 20 +
        ["Light Sleep"] * 120 +
        ["Deep Sleep"] * 90 +
        ["Light Sleep"] * 150 +
        ["Awake"] * 20
    )
    return {"Awake": 40, "Light Sleep": 270, "Deep Sleep": 90}

# Sleep Tips Based on Analysis
def display_sleep_tips(durations):
    """
    Display personalized tips based on the durations of sleep stages.
    """
    tips = []

    # Tips for Deep Sleep
    if durations.get("Deep Sleep", 0) < 60:
        tips.append("🔴 Your deep sleep is very low! This can affect muscle recovery and immune health. Try relaxing activities like meditation or avoid caffeine before bed.")
    elif durations.get("Deep Sleep", 0) > 120:
        tips.append("🟢 Great deep sleep duration! This supports recovery, memory consolidation, and overall health.")

    # Tips for Light Sleep
    if durations.get("Light Sleep", 0) > 300:
        tips.append("🟡 You spend too much time in light sleep. Reduce blue light exposure in the evening and maintain a consistent sleep schedule.")
    elif durations.get("Light Sleep", 0) < 200:
        tips.append("🔴 Your light sleep is insufficient. Light sleep is crucial for transitioning into restorative stages. Consider maintaining a consistent bedtime routine.")

    # Tips for Awake Time
    if durations.get("Awake", 0) > 45:
        tips.append("🟡 You spend a lot of time awake during the night. Ensure your sleeping environment is comfortable and minimize distractions.")
    elif durations.get("Awake", 0) < 30:
        tips.append("🟢 Minimal awake time! Your sleep efficiency is excellent.")

    # Additional General Tips
    tips.append("✅ Tip: Maintain a dark, quiet, and cool bedroom environment for optimal sleep quality.")
    tips.append("✅ Tip: Avoid heavy meals, alcohol, or caffeine close to bedtime to improve your sleep cycle.")

    # Display tips in a Google UI-like styled card
    html_tips = """
    <style>
        .card {
            font-family: 'Arial', sans-serif;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 16px auto;
            padding: 16px;
            max-width: 600px;
            background-color: #f9f9f9;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            font-size: 18px;
            border-radius: 6px 6px 0 0;
        }
        .content {
            padding: 16px;
        }
        .tip {
            margin: 8px 0;
            font-size: 14px;
        }
    </style>
    <div class="card">
        <div class="header">Personalized Sleep Tips</div>
        <div class="content">
    """
    for tip in tips:
        html_tips += f"<div class='tip'>{tip}</div>"
    html_tips += "</div></div>"
    display(HTML(html_tips))

# Main Execution
sleep_durations = generate_sleep_data()
display_sleep_tips(sleep_durations)

import pandas as pd
from IPython.display import display, HTML
import ipywidgets as widgets

# Simulated Sleep Data
def generate_sleep_data():
    """
    Generate default sleep stage data for an 8-hour period.
    """
    stages = (
        ["Awake"] * 20 +
        ["Light Sleep"] * 120 +
        ["Deep Sleep"] * 90 +
        ["Light Sleep"] * 150 +
        ["Awake"] * 20
    )
    time = pd.date_range("23:00", "07:00", periods=len(stages))
    return pd.DataFrame({"Time": time, "Stage": stages})

# Calculate Sleep Stage Durations
def calculate_sleep_durations(data):
    """
    Calculate the total duration of each sleep stage.
    """
    return data["Stage"].value_counts()

# Display Sleep Tips in Google Colab UI
def display_sleep_tips(durations):
    """
    Display personalized sleep tips in Google Colab using widgets.
    """
    tips = []

    # Conditions for sleep tips
    if durations.get("Deep Sleep", 0) < 60:
        tips.append("🔴 Your deep sleep is very low! Consider avoiding screens before bed or engaging in relaxation techniques to improve muscle recovery.")
    else:
        tips.append("🟢 Excellent deep sleep! Keep maintaining this for muscle recovery and memory consolidation.")

    if durations.get("Light Sleep", 0) > 300:
        tips.append("🟡 You have excessive light sleep. Ensure a consistent sleep schedule and reduce caffeine intake in the evening.")

    if durations.get("Awake", 0) > 45:
        tips.append("🟡 You spend a lot of time awake during sleep. Try mindfulness exercises or ensure a comfortable sleeping environment.")

    if durations.get("Deep Sleep", 0) > 120:
        tips.append("🟢 Fantastic! You have optimal deep sleep levels, which will support recovery and immune health.")

    if durations.get("Light Sleep", 0) < 200:
        tips.append("🔴 You have very little light sleep. This might indicate difficulty transitioning into deep sleep. Focus on relaxation before bed.")

    if durations.get("Awake", 0) < 30:
        tips.append("🟢 Minimal time awake during sleep! This indicates good sleep efficiency.")

    # Create a widget for tips display
    tips_box = widgets.VBox([
        widgets.HTML(value=f"<h3 style='color:#4CAF50;'>Personalized Sleep Tips</h3>")
    ])

    for tip in tips:
        tips_box.children += (widgets.HTML(value=f"<p>{tip}</p>"),)

    display(tips_box)

# Main Execution
sleep_data = generate_sleep_data()
durations = calculate_sleep_durations(sleep_data)
display_sleep_tips(durations)

"""# Changes Made:
Removed the Number of Awakenings metric from the analysis and visualization.
The Potential Insomnia determination is now based only on:
Sleep Efficiency being less than 85%.
Fragmentation Index being greater than 1.0.
This version focuses only on Sleep Efficiency and Fragmentation Index to determine potential insomnia. The rest of the logic and plotting remains unchanged.







"""

# Helper function for extracting hours from time string
def extract_hours(time_str):
    """
    Extracts hours from a time string in format 'X hrs Y mins'
    """
    hrs = float(time_str.split('hrs')[0])
    mins = float(time_str.split('hrs')[1].strip().split('mins')[0]) / 60
    return hrs + mins

def analyze_sleep_quality(results_in_hrs_mins, sleep_efficiency):
    """
    Analyzes sleep quality based on different sleep metrics and provides personalized recommendations.
    """
    # Extract sleep durations
    total_sleep = extract_hours(results_in_hrs_mins['Total Sleep Time'])
    deep_sleep = extract_hours(results_in_hrs_mins['Deep Sleep Time'])
    light_sleep = extract_hours(results_in_hrs_mins['Light Sleep Time'])
    awake_time = extract_hours(results_in_hrs_mins['Awake Time'])

    analysis = {
        'overall_quality': '',
        'muscle_recovery': '',
        'cognitive_function': '',
        'concerns': [],
        'recommendations': []
    }

    # Analyze total sleep duration
    if total_sleep >= 7 and total_sleep <= 9:
        analysis['overall_quality'] += "Your total sleep duration falls within the optimal range of 7-9 hours. "
    elif total_sleep < 7:
        analysis['overall_quality'] += "Your total sleep duration is below the recommended 7-9 hours. "
        analysis['concerns'].append("Insufficient total sleep time")
        analysis['recommendations'].append("Aim to get at least 7 hours of sleep per night")
    else:
        analysis['overall_quality'] += "Your sleep duration is longer than typical recommendations. "
        analysis['concerns'].append("Excessive sleep duration")

    # Analyze deep sleep
    deep_sleep_percentage = (deep_sleep / total_sleep) * 100
    if deep_sleep_percentage >= 20 and deep_sleep_percentage <= 25:
        analysis['muscle_recovery'] += "Your deep sleep percentage is optimal for muscle recovery and physical restoration. "
    elif deep_sleep_percentage < 20:
        analysis['muscle_recovery'] += "Your deep sleep percentage is lower than ideal for optimal muscle recovery. "
        analysis['concerns'].append("Insufficient deep sleep")
        analysis['recommendations'].append("Consider moderate exercise during the day")
    else:
        analysis['muscle_recovery'] += "You're getting abundant deep sleep, which is excellent for physical recovery. "

    # Analyze sleep efficiency
    sleep_eff = float(sleep_efficiency.strip('%'))
    if sleep_eff >= 85:
        analysis['overall_quality'] += "Your sleep efficiency is excellent. "
    elif sleep_eff >= 75:
        analysis['overall_quality'] += "Your sleep efficiency is acceptable but could be improved. "
        analysis['recommendations'].append("Try to maintain a consistent sleep schedule")
    else:
        analysis['overall_quality'] += "Your sleep efficiency needs improvement. "
        analysis['concerns'].append("Low sleep efficiency")
        analysis['recommendations'].append("Minimize time spent awake in bed")

    # Additional recommendations based on awake time
    if awake_time > 1:
        analysis['recommendations'].extend([
            "Create a relaxing bedtime routine",
            "Avoid screens 1 hour before bed",
            "Keep bedroom cool and dark"
        ])

    # Add sleep metrics to analysis for plotting
    analysis['metrics'] = {
        'total_sleep': total_sleep,
        'deep_sleep_percentage': deep_sleep_percentage,
        'sleep_efficiency': sleep_eff
    }

    return analysis

def generate_sleep_report(analysis):
    """
    Generates a formatted sleep report with recommendations.
    """
    report = """
🌙 Sleep Quality Analysis Report 🌙
==================================

Overall Sleep Quality:
{}

Muscle Recovery and Physical Restoration:
{}

Cognitive Function Impact:
{}
""".format(
        analysis['overall_quality'],
        analysis['muscle_recovery'],
        "Good sleep patterns support optimal cognitive function, memory consolidation, and mental clarity."
    )

    if analysis['concerns']:
        report += "\n⚠️ Areas for Improvement:\n"
        for concern in analysis['concerns']:
            report += f"- {concern}\n"

    report += "\n💡 Recommendations for Better Sleep:\n"
    for i, rec in enumerate(analysis['recommendations'], 1):
        report += f"{i}. {rec}\n"

    report += """
Additional Tips for Better Sleep:
1. Maintain a consistent sleep schedule
2. Create a comfortable sleep environment (18-20°C)
3. Limit caffeine intake after 2 PM
4. Get regular exercise, but not too close to bedtime
5. Practice relaxation techniques before bed
6. Use breathable, comfortable bedding
7. Consider using white noise or earplugs if needed
8. Avoid large meals close to bedtime
"""
    return report

def plot_sleep_quality_metrics(analysis):
    """
    Creates a visual representation of sleep quality metrics
    """
    plt.figure(figsize=(10, 6))

    # Define metrics to plot
    metrics = {
        'Total Sleep Score': min(100, (analysis['metrics']['total_sleep']/9) * 100),
        'Deep Sleep Score': min(100, (analysis['metrics']['deep_sleep_percentage']/25) * 100),
        'Sleep Efficiency': analysis['metrics']['sleep_efficiency']
    }

    # Create bar plot
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6'])

    # Customize the plot
    plt.title('Sleep Quality Metrics', pad=20)
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Run the analysis
analysis_results = analyze_sleep_quality(results_in_hrs_mins, results_in_hrs_mins['Sleep Efficiency (%)'])

# Generate and print the report
print(generate_sleep_report(analysis_results))

