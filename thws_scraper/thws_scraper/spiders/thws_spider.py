import scrapy
from urllib.parse import urljoin, urlparse
import fitz
import io
import re
from datetime import datetime
import unicodedata


class ThwsSpider(scrapy.Spider):
    name = "thws"
    allowed_domains = ["thws.de"]
    start_urls = ["https://www.thws.de/", "https://fiw.thws.de/"]

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the ThwsSpider with tracking for visited URLs and stats.
        """
        super().__init__(*args, **kwargs)
        self.visited: set[str] = set()
        self.stats: dict[str, int] = {
            "html": 0,
            "pdf": 0,
            "ical": 0,
            "errors": 0,
            "total": 0,
        }

    def parse(self, response: scrapy.http.Response) -> None:
        """
        Parse the response from the website. Handles HTML, PDF, and iCal responses.
        Skips real and soft 404 pages.

        Skips this:
        {"url": "https://fwi.thws.de/en/404", "type": "html", "title": "404 :: Fakultät Wirtschaftsingenieurwesen", ...}
        """
        url = self.normalize_url(response.url)

        if url in self.visited:
            return
        self.visited.add(url)

        # Skip hard 404s
        if response.status == 404:
            self.logger.debug(f"Skipping 404 (status): {url}")
            return

        # Extract title early for soft-404 detection
        page_title = response.css("title::text").get(default="").strip().lower()
        if "404" in page_title or "not found" in page_title:
            self.logger.debug(f"Skipping soft-404 (title): {url}")
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8").lower()

        if url.lower().endswith(".pdf") or "application/pdf" in content_type:
            yield from self.parse_pdf(response)
            return

        if url.lower().endswith(".ics") or "text/calendar" in content_type:
            yield from self.parse_ical(response)
            return

        # Extract HTML content
        main_selectors = response.css("div#main, main, [role=main]")
        if main_selectors:
            raw_text = "\n".join(main_selectors.css("::text").getall())
        else:
            raw_text = "\n".join(response.css("body ::text").getall())

        cleaned_text = self.clean_text(raw_text)

        # Skip if body looks like a known 404 message
        if (
            "diese seite existiert nicht" in cleaned_text.lower()
            or "this page does not exist" in cleaned_text.lower()
        ):
            self.logger.debug(f"Skipping soft-404 (body): {url}")
            return

        # Prefer headline for title, fallback to <title>
        headline = response.css("h1::text").get()
        title = (
            headline.strip()
            if headline
            else response.css("title::text").get(default="").strip()
        )

        # Try extracting date from .meta
        raw_meta = response.css("div.meta::text").get()
        date = None
        if raw_meta:
            date_match = re.search(r"\d{2}\.\d{2}\.\d{4}", raw_meta)
            if date_match:
                try:
                    parsed = datetime.strptime(date_match.group(), "%d.%m.%Y")
                    date = parsed.isoformat()
                except ValueError:
                    pass
        else:
            date = self.extract_date(response)

        self.stats["html"] += 1
        self.stats["total"] += 1

        yield {
            "url": url,
            "type": "html",
            "title": title,
            "text": cleaned_text,
            "date_scraped": datetime.utcnow().isoformat(),
            "date_updated": date,
        }

        for href in response.css("a::attr(href)").getall():
            next_url = urljoin(url, href)
            parsed = urlparse(next_url)
            if parsed.netloc.endswith("thws.de"):
                normalized_next = self.normalize_url(next_url)
                if normalized_next not in self.visited:
                    yield response.follow(next_url, callback=self.parse)

    def parse_pdf(self, response: scrapy.http.Response) -> None:
        """
        Parse PDF content from the response and yield scraped data.
        """
        url = self.normalize_url(response.url)
        try:
            pdf_bytes = response.body
            text = ""
            with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()

            cleaned_text = self.clean_text(text)

            self.stats["pdf"] += 1
            self.stats["total"] += 1

            yield {
                "url": url,
                "type": "pdf",
                "title": "",
                "text": cleaned_text,
                "date_scraped": datetime.utcnow().isoformat(),
                "date_updated": None,
            }

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.warning(f"PDF parsing failed for {url}: {e}")

    def parse_ical(self, response: scrapy.http.Response) -> None:
        """
        Parse iCal content from the response and yield scraped data.
        """
        url = self.normalize_url(response.url)
        try:
            text = response.text
            cleaned_text = self.clean_text(text)
            title = self.extract_ical_title(text)

            self.stats["ical"] += 1
            self.stats["total"] += 1

            yield {
                "url": url,
                "type": "ical",
                "title": title,
                "text": cleaned_text,
                "date_scraped": datetime.utcnow().isoformat(),
                "date_updated": None,
            }

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.warning(f"iCal parsing failed for {url}: {e}")

    def clean_text(self, text: str) -> str:
        """
        Clean text by normalizing Unicode and removing excess whitespace,
        while preserving \n and \n\n to retain paragraph structure for better chunking.
        """
        # Normalize to standard unicode (e.g., fancy quotes → straight quotes)
        # You can also use "NFC" or "NFKD" instead of "NFKC" depending on how aggressive you want to be.
        text = unicodedata.normalize("NFKC", text)

        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return self.deduplicate_lines("\n".join(lines))

    def deduplicate_lines(self, text: str) -> str:
        """
        Remove duplicate lines from text while preserving order.
        """
        seen: set[str] = set()
        unique_lines = []
        for line in text.splitlines():
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by removing query parameters and trailing slashes.
        """
        parsed = urlparse(url)
        return parsed._replace(query="").geturl().rstrip("/")

    def extract_date(self, response: scrapy.http.Response) -> str | None:
        """
        Extract a date from the response using common meta and text selectors.
        Attempts to parse and standardize the date format.

        Returns:
            A standardized ISO date string if parsing succeeds, else the raw date string,
            or None if no date is found.
        """
        selectors = [
            'meta[property="article:published_time"]::attr(content)',
            'meta[name="date"]::attr(content)',
            "time::text",
            ".date::text",
        ]
        for selector in selectors:
            date_str = response.css(selector).get()
            if date_str:
                date_str = date_str.strip()
                try:
                    # Attempt to parse to a standardized ISO format
                    parsed_date = datetime.fromisoformat(date_str)
                    return parsed_date.isoformat()
                except ValueError:
                    # If parsing fails, return the raw string
                    return date_str
        return None

    def extract_ical_title(self, text: str) -> str:
        """
        Extract the title from iCal content using a regex pattern.
        """
        match = re.search(r"SUMMARY:(.+)", text)
        return match.group(1).strip() if match else ""

    def closed(self, reason: str) -> None:
        """
        Log a summary of the crawling statistics upon spider closure.
        """
        self.logger.info("=== CRAWLING SUMMARY ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key.upper()}: {value}")
        self.logger.info(f"Spider closed because: {reason}")
