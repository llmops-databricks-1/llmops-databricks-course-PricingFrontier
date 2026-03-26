# Databricks notebook source
"""
Bronze layer ingestion: scrape actuarial GLM resources from the web
and upload them raw to Unity Catalog volumes.

Target volume: /Volumes/glam/bronze/raw_actuarial_docs/

Focus: GLM theory, actuarial best practices for fitting GLMs,
frequency-severity modelling, Tweedie, credibility, model selection,
and loss reserving with GLMs.

Sources:
  1. arXiv papers (tightly filtered actuarial GLM papers only)
  2. CAS monographs & study notes
  3. SOA research reports
  4. NAIC regulatory guidance on GLMs
  5. Academic papers (Tweedie, credibility-GLM, freq-sev, reserving)
  6. Lecture notes & textbook chapters
  7. Datasets (freMTPL2, CASdatasets, Schedule P)
  8. Reference pages (Wikipedia, IFoA, SSRN, ASTIN Bulletin)
"""

import json
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError as URLLibHTTPError

import arxiv
import requests
from bs4 import BeautifulSoup
from databricks.sdk import WorkspaceClient
from loguru import logger

# COMMAND ----------

VOLUME_PATH = "/Volumes/glam/bronze/raw_actuarial_docs"
SCRAPE_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
HEADERS = {"User-Agent": "GLAMBot/1.0 (actuarial-research)"}
TIMEOUT = 60

w = WorkspaceClient()

# COMMAND ----------

# --- Helpers ---


def upload_bytes(content: bytes, dest_path: str) -> None:
    w.files.upload(dest_path, BytesIO(content), overwrite=True)
    logger.info(f"Uploaded {dest_path} ({len(content):,} bytes)")


def upload_file(local_path: str, dest_path: str) -> None:
    with open(local_path, "rb") as f:
        w.files.upload(dest_path, f, overwrite=True)
    logger.info(f"Uploaded {dest_path}")


def save_metadata(records: list[dict], name: str) -> None:
    manifest = {
        "scrape_timestamp": SCRAPE_TS,
        "source": name,
        "record_count": len(records),
        "records": records,
    }
    payload = json.dumps(manifest, indent=2).encode()
    dest = f"{VOLUME_PATH}/metadata/{name}_{SCRAPE_TS}.json"
    upload_bytes(payload, dest)


def download_pdf(url: str, dest: str, name: str) -> dict | None:
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  FAILED {name}: {e}")
        return None
    upload_bytes(resp.content, dest)
    return {
        "name": name,
        "url": url,
        "volume_path": dest,
        "size_bytes": len(resp.content),
    }


def download_page(
    url: str, name: str, category: str
) -> dict | None:
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  FAILED {name}: {e}")
        return None

    safe = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
    )[:80]
    html_dest = f"{VOLUME_PATH}/html/{category}/{safe}.html"
    upload_bytes(resp.content, html_dest)

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    txt_dest = f"{VOLUME_PATH}/html/{category}/{safe}.txt"
    upload_bytes(text.encode("utf-8"), txt_dest)

    return {
        "name": name,
        "url": url,
        "category": category,
        "html_path": html_dest,
        "txt_path": txt_dest,
        "size_bytes": len(resp.content),
        "text_length": len(text),
    }


# Keywords that indicate a paper is actually about actuarial GLMs
GLM_KEYWORDS = {
    "generalized linear model", "generalised linear model",
    "glm", "tweedie", "poisson regression", "gamma regression",
    "frequency-severity", "frequency severity",
    "claim frequency", "claim severity", "pure premium",
    "insurance pricing", "insurance rating", "ratemaking",
    "loss reserving", "claims reserving", "chain ladder",
    "credibility", "tariff", "non-life insurance",
    "general insurance", "actuarial",
}


def is_relevant(result: arxiv.Result) -> bool:
    """Filter arXiv results to only actuarial GLM papers."""
    text = (result.title + " " + result.summary).lower()
    matches = sum(1 for kw in GLM_KEYWORDS if kw in text)
    return matches >= 2


# COMMAND ----------

# ============================================================
# 1. arXiv papers — tightly filtered to actuarial GLMs
# ============================================================

ARXIV_QUERIES = [
    '"generalized linear model" insurance pricing',
    '"generalized linear model" actuarial',
    "Tweedie compound Poisson insurance claims",
    "Poisson regression claim frequency insurance",
    "frequency severity model non-life insurance",
    '"loss reserving" "generalized linear model"',
    "credibility generalized linear model insurance",
    "GLM ratemaking insurance tariff",
    "overdispersion negative binomial insurance claims",
]
MAX_RESULTS_PER_QUERY = 20

logger.info("=== 1. Scraping arXiv ===")
seen_ids: set[str] = set()
arxiv_metadata: list[dict] = []

client = arxiv.Client(page_size=20, delay_seconds=5.0, num_retries=5)

for query in ARXIV_QUERIES:
    logger.info(f"  Query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS_PER_QUERY,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        for result in client.results(search):
            paper_id = result.entry_id.split("/")[-1]
            if paper_id in seen_ids:
                continue

            if not is_relevant(result):
                logger.debug(f"    Skipped (irrelevant): {result.title}")
                continue

            seen_ids.add(paper_id)
            safe_id = paper_id.replace("/", "_").replace(".", "_")
            dest = f"{VOLUME_PATH}/pdfs/arxiv/{safe_id}.pdf"

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_path = result.download_pdf(dirpath=tmpdir)
                    upload_file(str(pdf_path), dest)
            except (URLLibHTTPError, Exception) as e:
                logger.warning(f"    PDF download failed for {paper_id}: {e}")
                continue

            meta = {
                "id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.isoformat(),
                "abstract": result.summary,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "volume_path": dest,
            }
            arxiv_metadata.append(meta)
            logger.info(f"  [{len(arxiv_metadata)}] {result.title}")
    except arxiv.HTTPError as e:
        logger.warning(f"  arXiv rate limited on '{query}': {e}, waiting 30s...")
        time.sleep(30)
    time.sleep(3)

save_metadata(arxiv_metadata, "arxiv")
logger.info(f"arXiv complete: {len(arxiv_metadata)} papers")

# COMMAND ----------

# ============================================================
# 2. CAS monographs, study notes & papers
# ============================================================

logger.info("=== 2. CAS resources ===")

CAS_PDFS = [
    {
        "name": "CAS Monograph 5 - GLMs for Insurance Rating (Goldburd, Khare, Tevet)",
        "url": "https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf",
    },
    {
        "name": "CAS Monograph 3 - Stochastic Loss Reserving Using GLMs (Taylor, McGuire)",
        "url": "https://www.casact.org/sites/default/files/2021-03/7_Taylor.pdf",
    },
    {
        "name": "CAS Study Note - Practitioners Guide to GLMs (Anderson et al)",
        "url": "https://www.casact.org/sites/default/files/database/studynotes_anderson9.pdf",
    },
    {
        "name": "CAS Study Note - Practitioners Guide to GLMs (colour edition)",
        "url": "https://www.casact.org/sites/default/files/database/dpp_dpp04_04dpp1.pdf",
    },
    {
        "name": "Clark - GLMs in Practice Questions and Answers (CAS 2014)",
        "url": "https://www.casact.org/sites/default/files/2021-02/4_clark.pdf",
    },
    {
        "name": "CAS Educational Material - GLMs for Ratemaking",
        "url": "https://www.casact.org/sites/default/files/2021-02/glm-for-ratemaking.pdf",
    },
    {
        "name": "Loss Reserving with GLMs - A Case Study (Taylor, McGuire)",
        "url": "https://www.casact.org/pubs/dpp/dpp04/04dpp327.pdf",
    },
    {
        "name": "Generalized Mack Chain-Ladder with Robust Estimation (Sloma)",
        "url": "https://www.casact.org/sites/default/files/2021-07/Mack-Chain-Ladder-Sloma.pdf",
    },
    {
        "name": "Chain Ladder and Tweedie Distributed Claims Data (Taylor)",
        "url": "https://www.casact.org/sites/default/files/2021-07/Chain-Ladder-Tweedie-Taylor.pdf",
    },
    {
        "name": "Multivariate Buhlmann-Straub Credibility (Happ, Maier, Merz)",
        "url": "https://www.casact.org/sites/default/files/2021-08/Multivariate-Buhlmann-Straub-Happ-Maier-Merz.pdf",
    },
    {
        "name": "CAS Monograph 8 - Stochastic Loss Reserving (Meyers)",
        "url": "https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf",
    },
    {
        "name": "Machine Learning in Insurance (CAS Winter E-Forum 2022)",
        "url": "https://www.casact.org/sites/default/files/2022-03/01_Winter-Eforum-2022-ML_in_Insurance.pdf",
    },
]

cas_metadata: list[dict] = []
for resource in CAS_PDFS:
    filename = resource["url"].split("/")[-1]
    dest = f"{VOLUME_PATH}/pdfs/cas/{filename}"
    meta = download_pdf(resource["url"], dest, resource["name"])
    if meta:
        cas_metadata.append(meta)
        logger.info(f"  Downloaded {resource['name']}")

save_metadata(cas_metadata, "cas")
logger.info(f"CAS complete: {len(cas_metadata)} resources")

# COMMAND ----------

# ============================================================
# 3. SOA research reports
# ============================================================

logger.info("=== 3. SOA resources ===")

SOA_PDFS = [
    {
        "name": "SOA - Considerations for Predictive Modeling in Insurance (2019)",
        "url": "https://www.soa.org/globalassets/assets/files/resources/research-report/2019/considerations-predictive-modeling.pdf",
    },
    {
        "name": "SOA - Predictive Modeling for Life Insurance (Batty)",
        "url": "https://www.soa.org/globalassets/assets/files/research/projects/research-pred-mod-life-batty.pdf",
    },
    {
        "name": "SOA - Emerging Data Analytics Techniques with Actuarial Applications (2019)",
        "url": "https://www.soa.org/globalassets/assets/files/resources/research-report/2019/emerging-analytics-techniques-applications.pdf",
    },
    {
        "name": "SOA Study Note on GLMs for Exam Preparation",
        "url": "https://www.soa.org/globalassets/assets/files/edu/2020/spring/study-materials/stam-additional-study-note.pdf",
    },
]

soa_metadata: list[dict] = []
for resource in SOA_PDFS:
    filename = resource["url"].split("/")[-1]
    dest = f"{VOLUME_PATH}/pdfs/soa/{filename}"
    meta = download_pdf(resource["url"], dest, resource["name"])
    if meta:
        soa_metadata.append(meta)
        logger.info(f"  Downloaded {resource['name']}")

save_metadata(soa_metadata, "soa")
logger.info(f"SOA complete: {len(soa_metadata)} resources")

# COMMAND ----------

# ============================================================
# 4. NAIC regulatory guidance on GLMs
# ============================================================

logger.info("=== 4. NAIC regulatory resources ===")

NAIC_PDFS = [
    {
        "name": "NAIC - Regulatory Review of Predictive Models White Paper",
        "url": "https://content.naic.org/sites/default/files/CA-WP_1.pdf",
    },
    {
        "name": "NAIC - CASTF Predictive Model White Paper (2020)",
        "url": "https://content.naic.org/sites/default/files/inline-files/9-15%20CASTF%20-%20Predictive%20Model%20White%20Paper%209-09-2020_1.pdf",
    },
    {
        "name": "NAIC - Appendix B RGLM Update to White Paper",
        "url": "https://content.naic.org/sites/default/files/inline-files/Appendix%20B-RGLM%20Update%20to%20White%20Paper%20Update%20-%20%20with%20Redline.pdf",
    },
    {
        "name": "NAIC - Akur8 GLM Methodology and Regulatory Review (2022)",
        "url": "https://content.naic.org/sites/default/files/call_materials/Akur8%20GLM%20-%20Methodology%20and%20Regulatory%20review.pdf",
    },
]

naic_metadata: list[dict] = []
for resource in NAIC_PDFS:
    filename = resource["url"].split("/")[-1]
    dest = f"{VOLUME_PATH}/pdfs/naic/{filename}"
    meta = download_pdf(resource["url"], dest, resource["name"])
    if meta:
        naic_metadata.append(meta)
        logger.info(f"  Downloaded {resource['name']}")

save_metadata(naic_metadata, "naic")
logger.info(f"NAIC complete: {len(naic_metadata)} resources")

# COMMAND ----------

# ============================================================
# 5. Academic papers — GLM theory & actuarial practice
# ============================================================

logger.info("=== 5. Academic papers ===")

ACADEMIC_PDFS = [
    {
        "name": "Fitting Tweedies Compound Poisson Model to Insurance Claims (Smyth, Jorgensen) - ASTIN",
        "url": "https://www.casact.org/sites/default/files/old/astin_vol32no1_143.pdf",
    },
    {
        "name": "GLMs for Aggregate Claims - To Tweedie or Not",
        "url": "https://files01.core.ac.uk/download/pdf/211518053.pdf",
    },
    {
        "name": "Introducing Credibility Theory into GLMs for Ratemaking (Institut des Actuaires)",
        "url": "https://www.institutdesactuaires.com/docs/mem/090520123b7c578732a63f686535dcaa.pdf",
    },
    {
        "name": "Practitioners Guide to GLMs - 3rd Edition (Anderson et al 2007)",
        "url": "https://www.aktuarai.lt/wp-content/uploads/2018/06/Anderson_et_al_Edition_3.pdf",
    },
    {
        "name": "Comparison of GLM with ML for Predicting Loss Cost in Motor Insurance",
        "url": "https://www.open-access.bcu.ac.uk/15392/1/risks-12-00062-v2.pdf",
    },
    {
        "name": "Non-Life Insurance Pricing Using GAM (KTH Masters Thesis)",
        "url": "https://www.math.kth.se/matstat/seminarier/reports/M-exjobb15/150609a.pdf",
    },
    {
        "name": "Auto Insurance Premium Calculation Using GLMs (ScienceDirect)",
        "url": "https://www.sciencedirect.com/science/article/pii/S2212567115000593/pdf?md5=54b9012e51ca36c1c0a29f3af4090644&pid=1-s2.0-S2212567115000593-main.pdf",
    },
    {
        "name": "Stochastic Claims Reserving with Regression Models (Liverpool Thesis)",
        "url": "https://livrepository.liverpool.ac.uk/3028750/1/201048448_Set2018_edited_version.pdf",
    },
    {
        "name": "ASTIN Practical Model for Pricing Optimization in Car Insurance",
        "url": "https://actuaries.org/app/uploads/2025/07/ASTIN_Paper_Mayorga_Panama2017.pdf",
    },
    {
        "name": "Credibility Practice Note (American Academy of Actuaries 2008)",
        "url": "https://www.actuary.org/sites/default/files/files/publications/Practice_note_on_applying_credibility_theory_july2008.pdf",
    },
    {
        "name": "CAS Buhlmann Study Note on Credibility",
        "url": "https://thecasinstitute.org/wp-content/uploads/2019/01/Exam-3-Study-Note-Credibility01162019.pdf",
    },
    {
        "name": "Mahlers Guide to Advanced Ratemaking (sample)",
        "url": "https://www.actexmadriver.com/samples/Mahler_8C-MAH-17FSM_sample_4-24-17.pdf",
    },
    {
        "name": "Insurance Risk Classification Lecture Notes (Wuthrich, ETH Zurich)",
        "url": "https://people.math.ethz.ch/~wueth/Lectures/SAV_Weiterbildungskurs2017.pdf",
    },
    {
        "name": "P&C Pricing in the Age of Machine Learning (2024)",
        "url": "https://formacion.actuarios.org/wp-content/uploads/2024/05/2-PC-Pricing-in-the-Age-of-Machine-Learning-Jan-2024.pdf",
    },
    {
        "name": "Fitting Tweedies Compound Poisson Model - Cambridge Core (Smyth, Jorgensen)",
        "url": "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/DEF0B49F96FC015C7FBE076BC0A5C3AC/S051503610001299Xa.pdf/div-class-title-fitting-tweedie-s-compound-poisson-model-to-insurance-claims-data-dispersion-modelling-div.pdf",
    },
    {
        "name": "Neural Networks for Insurance Pricing - Benchmark Study (arXiv 2310.12671)",
        "url": "https://arxiv.org/pdf/2310.12671",
    },
    {
        "name": "Towards Explainability of ML Models in Insurance Pricing (arXiv 2003.10674)",
        "url": "https://arxiv.org/pdf/2003.10674",
    },
    {
        "name": "Comparison of Offset and Ratio Weighted Regressions in Tweedie (arXiv 2502.11788)",
        "url": "https://arxiv.org/pdf/2502.11788",
    },
]

academic_metadata: list[dict] = []
for resource in ACADEMIC_PDFS:
    filename = resource["url"].split("/")[-1][:100]
    safe = filename.replace("%20", "_").replace(" ", "_")
    dest = f"{VOLUME_PATH}/pdfs/academic/{safe}"
    if not dest.endswith(".pdf"):
        dest += ".pdf"
    meta = download_pdf(resource["url"], dest, resource["name"])
    if meta:
        academic_metadata.append(meta)
        logger.info(f"  Downloaded {resource['name']}")

save_metadata(academic_metadata, "academic")
logger.info(f"Academic complete: {len(academic_metadata)} papers")

# COMMAND ----------

# ============================================================
# 6. Lecture notes & textbook chapters
# ============================================================

logger.info("=== 6. Lecture notes & textbook chapters ===")

LECTURE_PDFS = [
    {
        "name": "Insurance Pricing Analytics - Lecture Sheets (Katrien Antonio, KU Leuven)",
        "url": "https://katrienantonio.github.io/PE-pricing-analytics/sheets/pricing_analytics_lecture_sheets_in_pdf.pdf",
    },
    {
        "name": "Buhlmann Credibility Lecture Notes (SMU)",
        "url": "http://www.mysmu.edu/faculty/yktse/NAM/NAM_S7.pdf",
    },
    {
        "name": "IFoA - Generalized Linear Models An Introduction",
        "url": "https://actuaries.org.uk/media/qemdewmk/glms-an-introduction.pdf",
    },
    {
        "name": "Jed Frees - Regression Modeling GLM Chapter (free PDF)",
        "url": "https://instruction.bus.wisc.edu/jfrees/jfreesbooks/Regression%20Modeling/BookWebDec2010/GLMChapter.pdf",
    },
]

LECTURE_PAGES = [
    {
        "name": "Loss Data Analytics - Open Actuarial Textbook (Credibility Chapter)",
        "url": "https://openacttexts.github.io/Loss-Data-Analytics/ChapCredibility.html",
    },
    {
        "name": "Open Actuarial Textbooks Project Home",
        "url": "https://openacttexts.github.io/",
    },
    {
        "name": "Frees Predictive Modeling Vol 1 Online Companion",
        "url": "https://instruction.bus.wisc.edu/jfrees/jfreesbooks/PredictiveModelingVol1/",
    },
    {
        "name": "Actuarial Data Science Tutorials (actuarialdatascience.org)",
        "url": "https://actuarialdatascience.org/",
    },
    {
        "name": "Wuthrich Papers Page (ETH Zurich RiskLab)",
        "url": "https://people.math.ethz.ch/~wueth/papers.html",
    },
    {
        "name": "Deutsche Aktuarvereinigung - Claim Frequency Modelling Theory",
        "url": "https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/",
    },
]

lecture_metadata: list[dict] = []

for resource in LECTURE_PDFS:
    filename = resource["url"].split("/")[-1]
    dest = f"{VOLUME_PATH}/pdfs/lectures/{filename}"
    meta = download_pdf(resource["url"], dest, resource["name"])
    if meta:
        meta["category"] = "lecture_pdf"
        lecture_metadata.append(meta)
        logger.info(f"  Downloaded {resource['name']}")

for page in LECTURE_PAGES:
    meta = download_page(page["url"], page["name"], "lectures")
    if meta:
        lecture_metadata.append(meta)
        logger.info(f"  Saved {page['name']}")

save_metadata(lecture_metadata, "lectures")
logger.info(f"Lectures complete: {len(lecture_metadata)} resources")

# COMMAND ----------

# ============================================================
# 7. Reference pages
# ============================================================

logger.info("=== 8. Reference pages ===")

REFERENCE_PAGES = [
    {
        "name": "Wikipedia - Generalized Linear Model",
        "url": "https://en.wikipedia.org/wiki/Generalized_linear_model",
    },
    {
        "name": "Wikipedia - Tweedie Distribution",
        "url": "https://en.wikipedia.org/wiki/Tweedie_distribution",
    },
    {
        "name": "Wikipedia - Exponential Dispersion Model",
        "url": "https://en.wikipedia.org/wiki/Exponential_dispersion_model",
    },
    {
        "name": "Wikipedia - Poisson Regression",
        "url": "https://en.wikipedia.org/wiki/Poisson_regression",
    },
    {
        "name": "Wikipedia - Gamma Distribution",
        "url": "https://en.wikipedia.org/wiki/Gamma_distribution",
    },
    {
        "name": "Wikipedia - Negative Binomial Distribution",
        "url": "https://en.wikipedia.org/wiki/Negative_binomial_distribution",
    },
    {
        "name": "Wikipedia - Buhlmann Model (Credibility)",
        "url": "https://en.wikipedia.org/wiki/B%C3%BChlmann_model",
    },
    {
        "name": "ASTIN Bulletin Journal (Cambridge Core)",
        "url": "https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa",
    },
    {
        "name": "IFoA - General Insurance Pricing Actuaries Considerations",
        "url": "https://www.actuaries.org.uk/practice-areas/general-insurance/research-working-parties/general-insurance-pricing-actuaries-considerations",
    },
    {
        "name": "IFoA - Advanced Pricing Techniques Working Party",
        "url": "https://www.actuaries.org.uk/practice-areas/general-insurance/disbanded-research-working-parties/advanced-pricing-techniques-general-insurance",
    },
    {
        "name": "IFoA - SP8 General Insurance Pricing Resources",
        "url": "https://actuaries.org.uk/studying/curriculum/general-insurance/resources-subject-sp8-general-insurance-pricing-principles",
    },
    {
        "name": "SOA - General Insurance Research Reports",
        "url": "https://www.soa.org/research/topics/general-ins-report-list/",
    },
    {
        "name": "CAS Monographs Landing Page",
        "url": "https://www.casact.org/publications-research/publications/flagship-publications/cas-monographs",
    },
    {
        "name": "CAS E-Forum - GLM for Dummies and Actuaries",
        "url": "https://eforum.casact.org/article/83925-glm-for-dummies-and-actuaries",
    },
    {
        "name": "Wuthrich & Merz - Statistical Foundations of Actuarial Learning (Springer)",
        "url": "https://link.springer.com/book/10.1007/978-3-031-12409-9",
    },
    {
        "name": "SSRN - Wuthrich Data Analytics for Non-Life Insurance Pricing",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2870308",
    },
    {
        "name": "SSRN - Wuthrich Statistical Methods for Insurance Pricing",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2676815",
    },
    {
        "name": "SSRN - Noll Salzmann Wuthrich French Motor TPL Case Study",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764",
    },
    {
        "name": "An Introduction to Statistical Learning (ISLR free textbook)",
        "url": "https://www.statlearning.com/",
    },
    {
        "name": "Freakonometrics - Arthur Charpentier Blog (GLM tag)",
        "url": "https://freakonometrics.hypotheses.org/tag/glm",
    },
    {
        "name": "Wuthrich - LocalGLMnet Architecture (SSRN)",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900350",
    },
    {
        "name": "MDPI Risks - GLM vs ML Motor Insurance Loss Cost",
        "url": "https://www.mdpi.com/2227-9091/12/4/62",
    },
    {
        "name": "CRAN Task View - Actuarial Science",
        "url": "https://cran.r-project.org/web/views/ActuarialScience.html",
    },
    {
        "name": "Actuaries Digital - Kaggle Insurance Competition Deep Dive",
        "url": "https://www.actuaries.digital/2021/10/06/kaggle-competition-modelling-of-claims-costs-and-a-deep-dive-into-the-winning-solution/",
    },
]

reference_metadata: list[dict] = []
for page in REFERENCE_PAGES:
    meta = download_page(page["url"], page["name"], "reference")
    if meta:
        reference_metadata.append(meta)
        logger.info(f"  Saved {page['name']}")

save_metadata(reference_metadata, "reference")
logger.info(f"Reference complete: {len(reference_metadata)} pages")

# COMMAND ----------

# ============================================================
# Summary
# ============================================================

counts = {
    "arxiv_papers": len(arxiv_metadata),
    "cas_resources": len(cas_metadata),
    "soa_reports": len(soa_metadata),
    "naic_regulatory": len(naic_metadata),
    "academic_papers": len(academic_metadata),
    "lecture_notes": len(lecture_metadata),
    "reference_pages": len(reference_metadata),
}
total = sum(counts.values())

logger.info("=" * 60)
logger.info(f"BRONZE INGESTION COMPLETE: {total} resources uploaded")
for source, count in counts.items():
    logger.info(f"  {source:20s}: {count}")
logger.info(f"Volume: {VOLUME_PATH}")
logger.info("=" * 60)

summary = {"scrape_timestamp": SCRAPE_TS, "total": total, **counts}
upload_bytes(
    json.dumps(summary, indent=2).encode(),
    f"{VOLUME_PATH}/metadata/summary_{SCRAPE_TS}.json",
)
