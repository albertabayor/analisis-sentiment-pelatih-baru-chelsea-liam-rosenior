"""
Sentiment Analysis Project - Chelsea Liam Rosenior Appointment

Utility functions for path handling and Google Colab compatibility.
"""

import sys
import os


def is_running_in_colab():
    """
    Check if the code is running in Google Colab.
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    return 'google.colab' in sys.modules


def get_project_root():
    """
    Get the project root directory.
    
    In Colab, returns the path in Google Drive.
    Locally, returns the project root directory.
    
    Returns:
        str: Path to project root directory
    """
    if is_running_in_colab():
        # In Colab, project is expected to be in Google Drive
        # User needs to mount Drive and upload project there
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Default path - user can customize this
        # Project should be uploaded to: MyDrive/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior
        return '/content/drive/MyDrive/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior'
    else:
        # Local execution - get script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)


def get_data_path(filename=None):
    """
    Get path to data directory or specific data file.
    
    Args:
        filename (str, optional): Name of data file. If None, returns data directory path.
    
    Returns:
        str: Full path to data directory or file
    """
    if filename:
        return os.path.join(get_project_root(), 'data', 'raw', filename)
    else:
        return os.path.join(get_project_root(), 'data', 'raw')


def get_processed_data_path(filename=None):
    """
    Get path to processed data directory or specific processed data file.
    
    Args:
        filename (str, optional): Name of processed data file. If None, returns processed data directory path.
    
    Returns:
        str: Full path to processed data directory or file
    """
    if filename:
        return os.path.join(get_project_root(), 'data', 'processed', filename)
    else:
        return os.path.join(get_project_root(), 'data', 'processed')


def get_outputs_path(subfolder=None):
    """
    Get path to outputs directory or specific subfolder.
    
    Args:
        subfolder (str, optional): Subfolder name (figures, tables, models, metrics). If None, returns outputs directory path.
    
    Returns:
        str: Full path to outputs directory or subfolder
    """
    if subfolder:
        return os.path.join(get_project_root(), 'outputs', subfolder)
    else:
        return os.path.join(get_project_root(), 'outputs')


def ensure_directories():
    """
    Create all necessary output directories if they don't exist.
    """
    directories = [
        get_processed_data_path(),
        get_outputs_path('figures'),
        get_outputs_path('tables'),
        get_outputs_path('models'),
        get_outputs_path('metrics'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"✓ Directories ensured:\n  - {chr(10).join([d.replace(get_project_root(), '.') for d in directories])}")


def install_requirements():
    """
    Install required packages in Colab environment.
    """
    if is_running_in_colab():
        print("Installing required packages...")
        import subprocess
        subprocess.run(['pip', 'install', '-q', 'pandas', 'numpy', 'scikit-learn', 'nltk', 
                        'vaderSentiment', 'langdetect', 'matplotlib', 'seaborn', 'wordcloud', 'plotly'],
                       check=True)
        print("✓ Packages installed successfully")
    else:
        print("Not running in Colab. Please install packages locally using:")
        print("pip install -r requirements.txt")


def download_nltk_data():
    """
    Download required NLTK data.
    """
    import nltk
    
    required_data = ['stopwords', 'wordnet', 'omw-1.4']
    
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
            print(f"✓ NLTK data already downloaded: {data}")
        except LookupError:
            print(f"Downloading NLTK data: {data}...")
            nltk.download(data)


def get_drive_folder_name():
    """
    Get the Google Drive folder name for this project.
    Users should upload their project to this folder in Google Drive.
    
    Returns:
        str: Expected Google Drive folder name
    """
    return 'analisis-sentiment-pelatih-baru-chelsea-liam-rosenior'


if __name__ == "__main__":
    # Test utility functions
    print(f"Running in Colab: {is_running_in_colab()}")
    print(f"Project root: {get_project_root()}")
    print(f"Data path: {get_data_path()}")
    print(f"Outputs path: {get_outputs_path()}")
    print(f"Drive folder name: {get_drive_folder_name()}")
