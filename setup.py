from setuptools import setup, find_packages

setup(
    name="helicopter-rul",
    version="1.0.0",
    description="Sikorsky S-76A/C+ Helicopter Engine Health Monitoring & Predictive Maintenance",
    author="Mridul Sharma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyarrow>=14.0.0",
        "openpyxl>=3.1.0",
        "streamlit>=1.28.0",
    ],
)
