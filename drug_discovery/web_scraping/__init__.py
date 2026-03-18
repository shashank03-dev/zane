"""
Web-Scale Data Ingestion Pipeline
Scrapes and processes biomedical literature and databases
"""

from .scraper import BiomedicalScraper, PubMedAPI, WebDataProcessor

__all__ = ['BiomedicalScraper', 'PubMedAPI', 'WebDataProcessor']
