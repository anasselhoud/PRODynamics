# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Tkinter
RUN apt-get update && apt-get install -y \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Run main-live.py when the container launches
CMD ["streamlit run", "app.py"]
