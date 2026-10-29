"""Deal data structures and the RSS scraper."""

from __future__ import annotations

import logging
import re
import time

import requests
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (price-estimate deal scanner)"}
TIMEOUT = 15


def clean_html(html: str) -> str:
    """Pull readable text out of an RSS summary snippet."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    node = soup.find("div", class_="snippet summary") or soup
    text = re.sub(r"<[^<]+?>", "", node.get_text(" ", strip=True))
    return re.sub(r"\s+", " ", text).strip()


class ScrapedDeal(BaseModel):
    """A raw deal from an RSS feed, enriched with text from its web page."""

    title: str
    summary: str
    url: str
    details: str = ""
    features: str = ""

    @classmethod
    def from_entry(cls, entry: dict, fetch_page: bool = True) -> ScrapedDeal:
        deal = cls(
            title=entry["title"][:100],
            summary=clean_html(entry.get("summary", "")),
            url=entry["links"][0]["href"],
        )
        if fetch_page:
            deal.enrich()
        return deal

    def enrich(self) -> None:
        """Fetch the deal page and split it into details / features. Failures are non-fatal."""
        from bs4 import BeautifulSoup

        try:
            html = requests.get(self.url, headers=HEADERS, timeout=TIMEOUT).content
        except requests.RequestException as error:
            log.warning("Could not fetch %s: %s", self.url, error)
            return
        section = BeautifulSoup(html, "html.parser").find("div", class_="content-section")
        if section is None:
            return
        content = section.get_text().replace("\nmore", "").replace("\n", " ")
        details, _, features = content.partition("Features")
        self.details, self.features = details.strip()[:500], features.strip()[:500]

    def describe(self) -> str:
        return (
            f"Title: {self.title}\nDetails: {self.details or self.summary}\n"
            f"Features: {self.features}\nURL: {self.url}"
        )

    @classmethod
    def fetch(cls, feeds: list[str] | None = None, per_feed: int = 10) -> list[ScrapedDeal]:
        import feedparser

        deals = []
        for feed_url in feeds or FEEDS:
            for entry in feedparser.parse(feed_url).entries[:per_feed]:
                try:
                    deals.append(cls.from_entry(entry))
                except (KeyError, IndexError) as error:
                    log.warning("Skipping malformed feed entry: %s", error)
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the "
        "item are much more important than why it's a good deal. Avoid mentioning discounts and "
        "coupons; focus on the item itself."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. If a deal is "
        "described as $100 off the usual $300 price, respond with 200."
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    deals: list[Deal] = Field(
        description="The 5 deals with the most detailed, high quality description and the clearest price."
    )


class Opportunity(BaseModel):
    """A deal plus our estimate of what it is really worth."""

    deal: Deal
    estimate: float
    discount: float
