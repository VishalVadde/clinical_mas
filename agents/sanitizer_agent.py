"""Sanitizer Agent - HIPAA PHI detection and removal using Microsoft Presidio.
Detects names, phones, emails, dates, SSNs, MRNs, account numbers, patient IDs.
Applies category-specific replacements and partial date masking.
"""
import re, time
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

mrn_recognizer = PatternRecognizer(
    supported_entity="MEDICAL_RECORD_NUMBER",
    patterns=[
        Pattern("MRN",  r"\bMRN[:\s#]+\d{6,10}\b", 0.9),
        Pattern("MR",   r"\bMR[:\s#]+\d{6,10}\b", 0.85),
    ]
)
account_recognizer = PatternRecognizer(
    supported_entity="ACCOUNT_NUMBER",
    patterns=[Pattern("ACCT", r"\b[Aa]ccount[:\s#]+\d{6,12}\b", 0.9)]
)
patient_id_recognizer = PatternRecognizer(
    supported_entity="PATIENT_ID",
    patterns=[Pattern("PID", r"\b[Pp]atient\s+ID[:\s#]+[\w\-]+\b", 0.9)]
)

class SanitizerAgent:
    REPLACEMENTS = {
        "PERSON": "[PATIENT]", "PHONE_NUMBER": "[PHONE]", "EMAIL_ADDRESS": "[EMAIL]",
        "LOCATION": "[LOCATION]", "US_SSN": "[SSN]", "US_DRIVER_LICENSE": "[ID]",
        "MEDICAL_RECORD_NUMBER": "[ID]", "ACCOUNT_NUMBER": "[ID]", "PATIENT_ID": "[ID]",
    }
    DATE_PATTERNS = [
        (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), r"\1/**/\3"),
        (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),     r"\1-\2-**"),
    ]

    def __init__(self, min_confidence=0.5):
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        })
        self.analyzer   = AnalyzerEngine(nlp_engine=provider.create_engine())
        self.anonymizer = AnonymizerEngine()
        for rec in [mrn_recognizer, account_recognizer, patient_id_recognizer]:
            self.analyzer.registry.add_recognizer(rec)
        self.min_confidence = min_confidence

    def _mask_dates(self, text):
        for pat, rep in self.DATE_PATTERNS:
            text = pat.sub(rep, text)
        return text

    def sanitize(self, text):
        t0 = time.perf_counter()
        text2 = self._mask_dates(text)
        results = self.analyzer.analyze(text=text2, language="en",
                                        score_threshold=self.min_confidence)
        ops = {k: OperatorConfig("replace", {"new_value": v})
               for k, v in self.REPLACEMENTS.items()}
        sanitized = (self.anonymizer.anonymize(text=text2, analyzer_results=results,
                                               operators=ops).text
                     if results else text2)
        return {
            "sanitized_text": sanitized, "original_text": text,
            "entities_found": [{"type": r.entity_type, "score": round(r.score, 3)}
                               for r in results],
            "n_entities": len(results),
            "latency_ms": round((time.perf_counter()-t0)*1000, 2),
            "phi_detected": len(results) > 0,
        }
