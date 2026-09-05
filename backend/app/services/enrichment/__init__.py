"""Enrichment passes that run over a merged result set: NER and clustering."""

from app.services.enrichment.clustering import ClusterResult, TopicClusterer
from app.services.enrichment.entities import (
    EntityExtractor,
    PatternEntityExtractor,
    SpacyEntityExtractor,
    build_entity_extractor,
)

__all__ = [
    "ClusterResult",
    "EntityExtractor",
    "PatternEntityExtractor",
    "SpacyEntityExtractor",
    "TopicClusterer",
    "build_entity_extractor",
]
