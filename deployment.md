# India Methane Emissions Monitor - Deployment Guide

This guide provides instructions for deploying the India Methane Emissions Monitor application in various environments, from development to production.

## Local Development Deployment

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Sufficient disk space for data files (approximately 5-10 GB depending on the size of your dataset)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/india-methane-monitor.git
   cd india-methane-monitor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify data directory structure:
   - Ensure the `data` folder contains state folders
   - Each state folder should contain district Excel files
   - The `DISTRICTs_Corrected` folder should contain shapefiles
   - `District_corrected_names.csv` should be in the root directory

5. Run the application in development mode:
   ```bash
   python app.py
   ```

6. Access the application at http://localhost:5000

## Production Deployment Options

### Option 1: Dedicated Server with Gunicorn and Nginx

#### Prerequisites

- Linux server (Ubuntu 20.04 or similar)
- Python 3.8 or higher
- Nginx
- Supervisor (optional, for process management)

#### Steps

1. Set up the server environment:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx supervisor
   ```

2. Create a dedicated user (optional but recommended):
   ```bash
   sudo adduser methane-app
   sudo su - methane-app
   ```

3. Clone and set up the application:
   ```bash
   git clone https://github.com/yourusername/india-methane-monitor.git
   cd india-methane-monitor
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn
   ```

4. Test Gunicorn deployment:
   ```bash
   gunicorn --bind 0.0.0.0:8000 app:app
   ```

5. Create a Supervisor configuration file:
   ```bash
   sudo nano /etc/supervisor/conf.d/methane-app.conf
   ```

   Add the following configuration:
   ```
   [program:methane-app]
   directory=/home/methane-app/india-methane-monitor
   command=/home/methane-app/india-methane-monitor/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
   autostart=true
   autorestart=true
   stderr_logfile=/var/log/methane-app/methane-app.err.log
   stdout_logfile=/var/log/methane-app/methane-app.out.log
   user=methane-app
   ```

6. Create log directory:
   ```bash
   sudo mkdir -p /var/log/methane-app
   sudo chown methane-app:methane-app /var/log/methane-app
   ```

7. Update Supervisor:
   ```bash
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start methane-app
   ```

8. Configure Nginx:
   ```bash
   sudo nano /etc/nginx/sites-available/methane-app
   ```

   Add the following configuration:
   ```
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

9. Enable the Nginx configuration:
   ```bash
   sudo ln -s /etc/nginx/sites-available/methane-app /etc/nginx/sites-enabled
   sudo nginx -t  # Test the configuration
   sudo systemctl restart nginx
   ```

10. Set up HTTPS with Let's Encrypt (recommended):
    ```bash
    sudo apt install certbot python3-certbot-nginx
    sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
    ```

### Option 2: Docker Deployment

#### Prerequisites

- Docker
- Docker Compose (optional, for easier management)

#### Steps

1. Create a Dockerfile in the project root:
   ```bash
   nano Dockerfile
   ```

   Add the following content:
   ```Dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN pip install gunicorn

   COPY . .

   EXPOSE 8000

   CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
   ```

2. Create a docker-compose.yml file (optional):
   ```bash
   nano docker-compose.yml
   ```

   Add the following content:
   ```yaml
   version: '3'

   services:
     web:
       build: .
       ports:
         - "8000:8000"
       volumes:
         - ./data:/app/data
         - ./DISTRICTs_Corrected:/app/DISTRICTs_Corrected
         - ./District_corrected_names.csv:/app/District_corrected_names.csv
       restart: always
   ```

3. Build and run the Docker container:
   ```bash
   # Using Docker directly
   docker build -t methane-app .
   docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/DISTRICTs_Corrected:/app/DISTRICTs_Corrected -v $(pwd)/District_corrected_names.csv:/app/District_corrected_names.csv methane-app

   # OR using Docker Compose
   docker-compose up -d
   ```

4. Configure Nginx as a reverse proxy (similar to Option 1, step 8)

### Option 3: Cloud Platform Deployment

#### AWS Elastic Beanstalk

1. Install the AWS EB CLI:
   ```bash
   pip install awsebcli
   ```

2. Create a .ebignore file to exclude large data files (they will be loaded separately):
   ```
   data/
   DISTRICTs_Corrected/
   ```

3. Initialize EB application:
   ```bash
   eb init -p python-3.8 methane-monitor
   ```

4. Create a requirements.txt file (if not already created)

5. Add a Procfile:
   ```
   web: gunicorn --bind 0.0.0.0:8080 app:app
   ```

6. Create EB environment:
   ```bash
   eb create methane-monitor-env
   ```

7. Upload data files to an S3 bucket and configure your application to download them on startup, or consider using an EFS mount

#### Google Cloud Platform (App Engine)

1. Create an app.yaml file:
   ```yaml
   runtime: python39
   instance_class: F4

   env_variables:
     BUCKET_NAME: "your-gcs-bucket-name"

   handlers:
   - url: /.*
     script: auto
   ```

2. Update your application to load data from Google Cloud Storage:
   ```python
   # Add to app.py
   from google.cloud import storage

   def download_data_from_gcs(bucket_name, prefix, local_dir):
       storage_client = storage.Client()
       bucket = storage_client.bucket(bucket_name)
       blobs = bucket.list_blobs(prefix=prefix)

       for blob in blobs:
           # Remove the prefix from the blob name
           rel_path = blob.name[len(prefix):].lstrip('/')
           local_path = os.path.join(local_dir, rel_path)
           
           # Create directories if they don't exist
           os.makedirs(os.path.dirname(local_path), exist_ok=True)
           
           # Download the blob
           blob.download_to_filename(local_path)
   ```

3. Upload data files to Google Cloud Storage

4. Deploy the application:
   ```bash
   gcloud app deploy
   ```

## Optimizing for Large Datasets

If your methane data is very large, consider these optimizations:

1. **Database Storage**: Convert Excel data to a database like PostgreSQL with PostGIS extension
   ```python
   # Example for PostgreSQL connection
   from sqlalchemy import create_engine
   import geopandas as gpd
   
   engine = create_engine('postgresql://username:password@localhost:5432/methane_db')
   gdf.to_postgis('methane_data', engine, if_exists='replace')
   ```

2. **Data Partitioning**: Partition data by year and state for faster queries
   ```sql
   -- Example PostgreSQL partitioning
   CREATE TABLE methane_readings (
       id SERIAL PRIMARY KEY,
       date DATE,
       state TEXT,
       district TEXT,
       latitude FLOAT,
       longitude FLOAT,
       methane_ppb FLOAT,
       geom GEOMETRY(Point, 4326)
   ) PARTITION BY RANGE (date);
   
   CREATE TABLE methane_readings_2014 PARTITION OF methane_readings
       FOR VALUES FROM ('2014-01-01') TO ('2015-01-01');
   
   CREATE TABLE methane_readings_2015 PARTITION OF methane_readings
       FOR VALUES FROM ('2015-01-01') TO ('2016-01-01');
   ```

3. **Caching Improvements**: Use Redis or Memcached for better caching performance
   ```bash
   pip install Flask-Caching[redis]
   ```
   
   Update Flask caching configuration:
   ```python
   cache = Cache(app, config={
       'CACHE_TYPE': 'redis',
       'CACHE_REDIS_URL': 'redis://localhost:6379/0'
   })
   ```

4. **Data Aggregation Pipeline**: Pre-aggregate data at different spatial and temporal scales to improve API performance

## Monitoring and Maintenance

1. **Application Monitoring**:
   - Set up Prometheus and Grafana for metrics
   - Configure logging with ELK stack (Elasticsearch, Logstash, Kibana)

2. **Backup Strategy**:
   - Regularly backup data files and the database
   - Create snapshots of server instances or containers

3. **Scaling Considerations**:
   - Implement horizontal scaling with multiple application servers and a load balancer
   - Consider using a content delivery network (CDN) for static assets

## Security Considerations

1. **API Security**:
   - Implement rate limiting to prevent abuse
   - Add API key authentication for sensitive endpoints

2. **Server Hardening**:
   - Keep all software updated
   - Implement firewall rules
   - Use SSH key authentication instead of passwords

3. **HTTPS**:
   - Always use HTTPS in production
   - Regularly renew SSL certificates
   - Configure proper SSL ciphers

## Troubleshooting

### Common Issues

1. **Application not starting**:
   - Check application logs
   - Verify Python version compatibility
   - Ensure all dependencies are installed correctly

2. **Data loading errors**:
   - Verify file paths and permissions
   - Check data file format consistency
   - Look for corrupt or invalid data

3. **Performance issues**:
   - Monitor memory usage - large datasets can cause memory problems
   - Check CPU utilization - processing intensive operations
   - Analyze database query performance

### Log Locations

- **Flask application logs**: Standard output or configured log file
- **Nginx logs**: `/var/log/nginx/`
- **Supervisor logs**: `/var/log/supervisor/` and `/var/log/methane-app/`
- **Docker logs**: `docker logs [container_id]` or `docker-compose logs`

## Contact for Support

For additional deployment assistance or troubleshooting help, contact:

- Email: support@example.com
- Issue tracker: https://github.com/yourusername/india-methane-monitor/issues