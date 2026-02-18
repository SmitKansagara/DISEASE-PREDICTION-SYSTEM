FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY zip/ ./zip/

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Create streamlit config file
RUN echo "\
[server]\n\
port = 8501\n\
headless = true\n\
runOnSave = true\n\
" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "zip/app.py", "--logger.level=error"]
