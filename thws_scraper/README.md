# 🕷️ THWS Scraper

This project contains a robust and feature-rich Scrapy-based web scraper designed to crawl THWS (University of Applied Sciences Würzburg-Schweinfurt) websites. It intelligently handles various content types, stores data efficiently in MongoDB, and provides real-time monitoring capabilities.

---

## ✨ Key Features

- **Multi-Content Type Parsing:**
    - **HTML:** Parses standard web pages, extracting main content and discovering new links.
    - **PDF:** Downloads and stores PDF documents.
    - **iCal:** Parses `.ics` files to extract structured event data.
- **Efficient MongoDB Storage:**
    - **GridFS for Large Files:** Automatically uses GridFS to store large files (e.g., PDFs), keeping the main database documents lean and performant.
    - **Direct Embedding:** Smaller files are embedded directly within MongoDB documents for faster access.
    - **Indexed Collections:** Creates unique indexes on the `url` field to prevent duplicate entries and ensure data integrity.
- **Live Monitoring & Statistics:**
    - **Web Dashboard:** A built-in, lightweight web server provides a live dashboard at `http://localhost:7000/live` to monitor crawl progress in real-time.
    - **JSON API:** Exposes raw statistics via a JSON endpoint at `http://localhost:7000/stats`.
    - **Detailed CSV Exports:** Automatically generates a CSV summary of the crawl statistics upon completion.
- **Intelligent Crawling & Error Handling:**
    - **Soft Error Detection:** Identifies and ignores "soft 404" pages (e.g., pages that return a 200 OK status but contain messages like "Seite nicht gefunden").
    - **Configurable URL Filtering:** Allows you to define URL patterns to ignore during a crawl, avoiding irrelevant content.
    - **Robust Retries:** Automatically retries failed requests for common transient HTTP errors (e.g., 500, 502, 503).
- **Containerized & Reproducible:**
    - **Docker Compose:** The entire stack (Scraper, MongoDB, Mongo Express) is managed with Docker Compose for easy, one-command setup.
    - **Environment-based Configuration:** All sensitive data and settings are managed via a `.env` file, keeping credentials out of the codebase.

---

## 🏗️ System Architecture

The scraper is designed as a multi-container Docker application:

1.  **`thws_scraper` Service:**
    - The core Scrapy spider that crawls the target websites.
    - It runs a lightweight `http.server` in a separate thread to serve the live stats dashboard.
    - It connects to the `mongodb` service to store the data it collects.

2.  **`mongodb` Service:**
    - A standard MongoDB instance that persists all scraped data.
    - It uses a Docker volume (`mongodb_data`) to ensure data is not lost when the container is stopped or restarted.

3.  **`mongo-express` Service:**
    - A web-based administration interface for MongoDB, accessible at `http://localhost:8081`.
    - It allows you to easily browse the `pages` and `files` collections to inspect the scraped data.

---

## 🔧 Configuration

Create a `.env` file in the project's root directory to configure the application. This file is ignored by Docker and Git to protect your credentials.

```env
# .env
MONGO_USER=scraper
MONGO_PASS=password
MONGO_DB=askthws_scraper
```

---

## 🛠 Usage

### 1. Build and Start the Services

From the project's root directory, run:

```bash
docker-compose up -d --build
```

This command will build the scraper image, pull the necessary MongoDB images, and start all services in the background.

### 2. Monitor the Crawl

- **Live HTML Dashboard:**
  Open your browser to [http://localhost:7000/live](http://localhost:7000/live) to see a real-time table of crawl statistics per subdomain.

- **Scraper Logs:**
  To view the detailed, JSON-formatted logs from the scraper, run:
  ```bash
  docker-compose logs -f scraper
  ```

- **Database UI:**
  To browse the MongoDB database, open [http://localhost:8081](http://localhost:8081) in your browser.

### 3. Data Management

#### Exporting Data

The project includes a convenient script to back up the entire scraper database from the running MongoDB container.

```bash
./dump_mongo.sh
```

This creates a compressed archive (`askthws_scraper_backup_*.gz`) in the current directory.

#### Importing Data

You can restore a backup into the container. This is useful for seeding a new environment or restoring a previous state.

```bash
# Replace <BACKUP_FILE_NAME.gz> with your backup file
docker exec -i mongodb mongorestore \
    --username scraper \
    --password password \
    --authenticationDatabase "admin" \
    --archive \
    --gzip \
    --drop < ./<BACKUP_FILE_NAME.gz>
```

**Note:** The `--drop` flag will delete existing collections before restoring. Remove it if you need to merge data.

### 4. Analyzing Scraped Data

A Python script is provided to generate a detailed statistical report from the data stored in MongoDB. This can be run from your local machine if you have Python and the required packages installed, or inside the scraper container.

```bash
# Ensure you have a .env file with credentials
python3 mongo_stats.py
```

This script provides insights into:
- Content types and language distribution.
- Scraped items per day.
- HTTP status code distribution.
- Content freshness and metadata completeness.