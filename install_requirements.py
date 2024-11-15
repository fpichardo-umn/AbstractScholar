import subprocess
import sys

def install_requirements():
    """Install required packages using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        # Install NLTK data
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        print("Successfully installed all requirements.")
    except Exception as e:
        print(f"Error installing requirements: {e}")

if __name__ == "__main__":
    install_requirements()
