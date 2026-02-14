"""Status-related service wrappers."""

from pdf_compressor.services import compression_service


def get_status(job_id: str):
    return compression_service.get_status(job_id)


def job_check_pdfco():
    return compression_service.job_check_pdfco()


def diagnose(job_id: str):
    return compression_service.diagnose(job_id)
