"""
Document Validator
==================
Production document validation and quality assessment for RAG pipeline.
"""

import logging
import re
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Document validation strictness levels."""
    LENIENT = "lenient"  # Allow most documents
    STANDARD = "standard"  # Reasonable quality checks
    STRICT = "strict"  # High quality requirements


class QualityIssue(Enum):
    """Types of quality issues."""
    EMPTY_CONTENT = "empty_content"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOW_TEXT_DENSITY = "low_text_density"
    EXCESSIVE_SPECIAL_CHARS = "excessive_special_chars"
    ENCODING_ISSUES = "encoding_issues"
    DUPLICATE_CONTENT = "duplicate_content"
    BOILERPLATE_HEAVY = "boilerplate_heavy"
    POOR_STRUCTURE = "poor_structure"
    LANGUAGE_MIXED = "language_mixed"


@dataclass
class ValidationConfig:
    """Configuration for document validation."""
    level: ValidationLevel = ValidationLevel.STANDARD
    min_content_length: int = 50
    max_content_length: int = 100000
    min_word_count: int = 10
    max_word_count: int = 50000
    min_text_density: float = 0.3  # Ratio of alphanumeric chars
    max_special_char_ratio: float = 0.3
    max_duplicate_ratio: float = 0.5
    min_unique_words_ratio: float = 0.1
    detect_language: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    issues: List[QualityIssue]
    warnings: List[str]
    metrics: Dict[str, Any]
    suggestions: List[str]


@dataclass
class DocumentMetrics:
    """Metrics extracted from document content."""
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    unique_word_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    text_density: float = 0.0
    special_char_ratio: float = 0.0
    numeric_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    content_hash: str = ""


class DocumentValidator:
    """
    Validates documents for quality and suitability for RAG.

    Features:
    - Content length validation
    - Text density analysis
    - Encoding detection
    - Duplicate detection
    - Structure analysis
    - Quality scoring
    """

    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        r"copyright\s+\d{4}",
        r"all\s+rights\s+reserved",
        r"terms\s+of\s+(service|use)",
        r"privacy\s+policy",
        r"cookie\s+policy",
        r"page\s+\d+\s+of\s+\d+",
        r"confidential",
        r"do\s+not\s+distribute",
    ]

    # Gibberish detection patterns
    GIBBERISH_PATTERNS = [
        r"[a-z]{20,}",  # Very long words without spaces
        r"(.)\1{5,}",  # Repeated characters
        r"[^a-zA-Z0-9\s]{10,}",  # Long sequences of special chars
    ]

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._seen_hashes: set = set()

    def validate(self, document: Document) -> ValidationResult:
        """
        Validate a single document.

        Args:
            document: Document to validate

        Returns:
            ValidationResult with quality assessment
        """
        content = document.page_content
        issues = []
        warnings = []
        suggestions = []

        # Extract metrics
        metrics = self._extract_metrics(content)

        # Check empty content
        if not content or not content.strip():
            issues.append(QualityIssue.EMPTY_CONTENT)
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=issues,
                warnings=warnings,
                metrics=vars(metrics),
                suggestions=["Document has no content"]
            )

        # Length checks
        if metrics.char_count < self.config.min_content_length:
            issues.append(QualityIssue.TOO_SHORT)
            suggestions.append(f"Content too short ({metrics.char_count} chars). Minimum: {self.config.min_content_length}")

        if metrics.char_count > self.config.max_content_length:
            issues.append(QualityIssue.TOO_LONG)
            suggestions.append(f"Content too long ({metrics.char_count} chars). Consider splitting.")

        # Word count checks
        if metrics.word_count < self.config.min_word_count:
            issues.append(QualityIssue.TOO_SHORT)
            warnings.append(f"Low word count: {metrics.word_count}")

        # Text density check
        if metrics.text_density < self.config.min_text_density:
            issues.append(QualityIssue.LOW_TEXT_DENSITY)
            suggestions.append("Document has low text density. May contain too many special characters or formatting.")

        # Special character check
        if metrics.special_char_ratio > self.config.max_special_char_ratio:
            issues.append(QualityIssue.EXCESSIVE_SPECIAL_CHARS)
            warnings.append(f"High special character ratio: {metrics.special_char_ratio:.2f}")

        # Duplicate content check
        if self._is_duplicate(metrics.content_hash):
            issues.append(QualityIssue.DUPLICATE_CONTENT)
            warnings.append("Duplicate or near-duplicate content detected")

        # Check for encoding issues
        if self._has_encoding_issues(content):
            issues.append(QualityIssue.ENCODING_ISSUES)
            suggestions.append("Document may have encoding issues. Check character encoding.")

        # Check for excessive boilerplate
        boilerplate_score = self._calculate_boilerplate_score(content)
        if boilerplate_score > 0.3:
            issues.append(QualityIssue.BOILERPLATE_HEAVY)
            warnings.append(f"High boilerplate content detected ({boilerplate_score:.1%})")

        # Check structure
        structure_score = self._assess_structure(content, metrics)
        if structure_score < 0.3:
            issues.append(QualityIssue.POOR_STRUCTURE)
            suggestions.append("Document has poor structure. Consider improving formatting.")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            metrics, issues, boilerplate_score, structure_score
        )

        # Determine if valid based on level
        is_valid = self._is_valid_for_level(issues, quality_score)

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            metrics=vars(metrics),
            suggestions=suggestions
        )

    def validate_batch(self, documents: List[Document]) -> Tuple[List[Document], List[ValidationResult]]:
        """
        Validate multiple documents and return valid ones.

        Args:
            documents: List of documents to validate

        Returns:
            Tuple of (valid documents, all validation results)
        """
        results = []
        valid_docs = []

        for doc in documents:
            result = self.validate(doc)
            results.append(result)

            if result.is_valid:
                valid_docs.append(doc)
            else:
                logger.warning(
                    f"Document failed validation: {doc.metadata.get('source', 'unknown')} - "
                    f"Issues: {[i.value for i in result.issues]}"
                )

        logger.info(f"Validated {len(documents)} documents. {len(valid_docs)} passed.")
        return valid_docs, results

    def _extract_metrics(self, content: str) -> DocumentMetrics:
        """Extract metrics from document content."""
        # Basic counts
        char_count = len(content)
        words = content.split()
        word_count = len(words)

        # Unique words
        unique_words = set(w.lower() for w in words if w.isalnum())
        unique_word_count = len(unique_words)

        # Sentences (simple heuristic)
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])

        # Paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraph_count = len([p for p in paragraphs if p.strip()])

        # Average lengths
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Character type ratios
        alphanumeric = sum(1 for c in content if c.isalnum())
        special = sum(1 for c in content if not c.isalnum() and not c.isspace())
        numeric = sum(1 for c in content if c.isdigit())
        uppercase = sum(1 for c in content if c.isupper())

        text_density = alphanumeric / max(char_count, 1)
        special_char_ratio = special / max(char_count, 1)
        numeric_ratio = numeric / max(char_count, 1)
        uppercase_ratio = uppercase / max(alphanumeric, 1)

        # Content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()

        return DocumentMetrics(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            unique_word_count=unique_word_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            text_density=text_density,
            special_char_ratio=special_char_ratio,
            numeric_ratio=numeric_ratio,
            uppercase_ratio=uppercase_ratio,
            content_hash=content_hash,
        )

    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if content is a duplicate."""
        if content_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(content_hash)
        return False

    def _has_encoding_issues(self, content: str) -> bool:
        """Check for common encoding issues."""
        # Check for replacement characters
        if '\ufffd' in content:
            return True

        # Check for common mojibake patterns
        mojibake_patterns = [
            r'â€™',  # Common UTF-8 decode issue
            r'Ã©',
            r'Ã¨',
            r'â€"',
            r'â€œ',
        ]
        for pattern in mojibake_patterns:
            if pattern in content:
                return True

        return False

    def _calculate_boilerplate_score(self, content: str) -> float:
        """Calculate ratio of boilerplate content."""
        content_lower = content.lower()
        boilerplate_matches = 0

        for pattern in self.BOILERPLATE_PATTERNS:
            if re.search(pattern, content_lower):
                boilerplate_matches += 1

        return boilerplate_matches / len(self.BOILERPLATE_PATTERNS)

    def _assess_structure(self, content: str, metrics: DocumentMetrics) -> float:
        """Assess document structure quality."""
        scores = []

        # Paragraph structure
        if metrics.paragraph_count > 1:
            scores.append(1.0)
        elif metrics.sentence_count > 3:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Sentence length variation
        if 10 < metrics.avg_sentence_length < 30:
            scores.append(1.0)
        elif 5 < metrics.avg_sentence_length < 50:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Word diversity
        if metrics.word_count > 0:
            diversity = metrics.unique_word_count / metrics.word_count
            if diversity > self.config.min_unique_words_ratio:
                scores.append(1.0)
            else:
                scores.append(diversity / self.config.min_unique_words_ratio)
        else:
            scores.append(0.0)

        # Check for gibberish
        gibberish_found = 0
        for pattern in self.GIBBERISH_PATTERNS:
            if re.search(pattern, content):
                gibberish_found += 1

        if gibberish_found == 0:
            scores.append(1.0)
        else:
            scores.append(max(0, 1 - gibberish_found * 0.3))

        return sum(scores) / len(scores)

    def _calculate_quality_score(
        self,
        metrics: DocumentMetrics,
        issues: List[QualityIssue],
        boilerplate_score: float,
        structure_score: float
    ) -> float:
        """Calculate overall quality score."""
        # Start with base score
        score = 1.0

        # Deduct for issues
        issue_penalties = {
            QualityIssue.EMPTY_CONTENT: 1.0,
            QualityIssue.TOO_SHORT: 0.3,
            QualityIssue.TOO_LONG: 0.1,
            QualityIssue.LOW_TEXT_DENSITY: 0.2,
            QualityIssue.EXCESSIVE_SPECIAL_CHARS: 0.2,
            QualityIssue.ENCODING_ISSUES: 0.3,
            QualityIssue.DUPLICATE_CONTENT: 0.5,
            QualityIssue.BOILERPLATE_HEAVY: 0.2,
            QualityIssue.POOR_STRUCTURE: 0.2,
            QualityIssue.LANGUAGE_MIXED: 0.1,
        }

        for issue in issues:
            score -= issue_penalties.get(issue, 0.1)

        # Factor in structure and boilerplate
        score *= (1 - boilerplate_score * 0.5)
        score *= (0.5 + structure_score * 0.5)

        # Factor in text density
        density_factor = min(metrics.text_density / self.config.min_text_density, 1.0)
        score *= (0.7 + density_factor * 0.3)

        return max(0.0, min(1.0, score))

    def _is_valid_for_level(self, issues: List[QualityIssue], quality_score: float) -> bool:
        """Determine if document is valid based on validation level."""
        critical_issues = {
            QualityIssue.EMPTY_CONTENT,
            QualityIssue.ENCODING_ISSUES,
        }

        # Critical issues always fail
        if any(issue in critical_issues for issue in issues):
            return False

        if self.config.level == ValidationLevel.LENIENT:
            return quality_score >= 0.2 and QualityIssue.EMPTY_CONTENT not in issues

        elif self.config.level == ValidationLevel.STANDARD:
            return quality_score >= 0.4 and len(issues) <= 3

        else:  # STRICT
            return quality_score >= 0.6 and len(issues) <= 1

    def clear_duplicate_cache(self):
        """Clear the duplicate detection cache."""
        self._seen_hashes.clear()


class DocumentPreprocessor:
    """
    Preprocesses documents before chunking.

    Features:
    - Content cleaning
    - Encoding normalization
    - Whitespace normalization
    - Special character handling
    - Boilerplate removal
    """

    def __init__(self):
        self._boilerplate_patterns = [
            re.compile(r"copyright\s+\d{4}.*?(?=\n|\Z)", re.IGNORECASE),
            re.compile(r"all\s+rights\s+reserved.*?(?=\n|\Z)", re.IGNORECASE),
            re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
            re.compile(r"confidential.*?(?=\n|\Z)", re.IGNORECASE),
        ]

    def preprocess(self, document: Document) -> Document:
        """
        Preprocess a document for better RAG performance.

        Args:
            document: Document to preprocess

        Returns:
            Preprocessed document
        """
        content = document.page_content

        # Normalize encoding
        content = self._normalize_encoding(content)

        # Clean content
        content = self._clean_content(content)

        # Remove boilerplate
        content = self._remove_boilerplate(content)

        # Normalize whitespace
        content = self._normalize_whitespace(content)

        return Document(
            page_content=content,
            metadata={
                **document.metadata,
                "preprocessed": True,
                "original_length": len(document.page_content),
                "processed_length": len(content),
            }
        )

    def preprocess_batch(self, documents: List[Document]) -> List[Document]:
        """Preprocess multiple documents."""
        return [self.preprocess(doc) for doc in documents]

    def _normalize_encoding(self, content: str) -> str:
        """Fix common encoding issues."""
        # Replace common mojibake
        replacements = {
            'â€™': "'",
            'â€"': "—",
            'â€œ': '"',
            'â€': '"',
            'Ã©': 'é',
            'Ã¨': 'è',
            '\ufffd': '',  # Remove replacement character
        }

        for old, new in replacements.items():
            content = content.replace(old, new)

        return content

    def _clean_content(self, content: str) -> str:
        """Clean document content."""
        # Remove null bytes
        content = content.replace('\x00', '')

        # Remove control characters except newlines and tabs
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)

        # Fix broken sentences (single newlines in middle of sentences)
        content = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', content)

        return content

    def _remove_boilerplate(self, content: str) -> str:
        """Remove common boilerplate text."""
        for pattern in self._boilerplate_patterns:
            content = pattern.sub('', content)

        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace in content."""
        # Replace multiple spaces with single space
        content = re.sub(r'[ \t]+', ' ', content)

        # Normalize line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Remove trailing whitespace
        content = '\n'.join(line.rstrip() for line in content.split('\n'))

        return content.strip()


# Convenience function
def validate_document(
    document: Document,
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """
    Validate a single document.

    Args:
        document: Document to validate
        level: Validation strictness level

    Returns:
        ValidationResult
    """
    config = ValidationConfig(level=level)
    validator = DocumentValidator(config)
    return validator.validate(document)


def preprocess_document(document: Document) -> Document:
    """
    Preprocess a document for RAG.

    Args:
        document: Document to preprocess

    Returns:
        Preprocessed document
    """
    preprocessor = DocumentPreprocessor()
    return preprocessor.preprocess(document)
